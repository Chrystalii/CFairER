import numpy as np
from sklearn.metrics import roc_auc_score
from torch import multiprocessing as mp
import torch

def get_score(u_es, i_es):
    u_es = u_es[:, :]

    score_matrix = torch.matmul(u_es, i_es.t())

    return score_matrix

def cal_ndcg(topk, test_set, num_pos, k):
    n = min(num_pos, k)
    nrange = np.arange(n) + 2
    idcg = np.sum(1 / np.log2(nrange))

    dcg = 0
    for i, s in enumerate(topk):
        if s in test_set:
            dcg += 1 / np.log2(i + 2)

    ndcg = dcg / idcg

    return ndcg

# def cal_gini(topk, data):
#     long_tails = set(data.long_tail_items)
#     head_tails = set(data.head_tail_items)


def test_model(u_es, i_es, ks, data):

    ks = eval(ks)

    gt = data.interaction_dict

    n_users = data.n_users
    n_test_users = len(gt)

    n_k = len(ks)
    result = {
        # "precision": np.zeros(n_k),
        # "recall": np.zeros(n_k),
        "ndcg": np.zeros(n_k),
        "hit_ratio": np.zeros(n_k),
        "head-tail_rate": np.zeros(n_k),
        'gini': np.zeros(n_k)
    }

    long_tails = set(data.head_tail_items)

    score_matrix = get_score(u_es, i_es)
    for i, k in enumerate(ks):
        # precision, recall, \
        ndcg, hr, lt, gini = 0, 0, 0, 0
        _, topk_index = torch.topk(score_matrix, k)
        topk_index = topk_index.cpu().numpy()

        for u in range(0, n_users):
            if u in gt.keys():
                gt_pos = gt[u]

                topk = topk_index[u]
                num_pos = len(gt_pos)

                topk_set = set(topk)
                test_set = set(gt_pos)
                num_hit = len(topk_set & test_set)
                long_tail_items = len(topk_set & long_tails)
                gini_u = compute_gini(topk_index[u])

                # precision += num_hit / k
                # recall += num_hit / num_pos
                hr += 1 if num_hit > 0 else 0
                ndcg += cal_ndcg(topk, test_set, num_pos, k)
                lt += long_tail_items/k
                gini += gini_u

            else:
                continue

        # result["precision"][i] += precision / n_test_users
        # result["recall"][i] += recall / n_test_users
        result["ndcg"][i] += ndcg / n_test_users
        result["hit_ratio"][i] += hr / n_test_users
        result["head-tail_rate"][i] += lt/n_test_users
        result["gini"][i] += gini/n_test_users

    return result

def get_row_indices(row, interactions):
    start = interactions.indptr[row]
    end = interactions.indptr[row + 1]
    return interactions.indices[start:end]


def auc(model, interactions, num_workers=1):
    aucs = []
    processes = []
    n_users = interactions.n_users
    mp_batch = int(np.ceil(n_users / num_workers))

    queue = mp.Queue()
    rows = np.arange(n_users)
    np.random.shuffle(rows)
    for rank in range(num_workers):
        start = rank * mp_batch
        end = np.min((start + mp_batch,  n_users))
        p = mp.Process(target=batch_auc,
                       args=(queue, rows[start:end], interactions, model))
        p.start()
        processes.append(p)

    while True:
        is_alive = False
        for p in processes:
            if p.is_alive():
                is_alive = True
                break
        if not is_alive and queue.empty():
            break

        while not queue.empty():
            aucs.append(queue.get())

    queue.close()
    for p in processes:
        p.join()
    return np.mean(aucs)


def batch_auc(queue, rows, interactions, model):
    n_items = interactions.n_items
    items = torch.arange(0, n_items).long()
    users_init = torch.ones(n_items).long()
    for row in rows:
        row = int(row)
        users = users_init.fill_(row)

        preds = model.predict(users, items)
        actuals = get_row_indices(row, interactions)

        if len(actuals) == 0:
            continue
        y_test = np.zeros(n_items)
        y_test[actuals] = 1
        queue.put(roc_auc_score(y_test, preds.data.numpy()))


def patk(model, interactions, num_workers=1, k=5):
    patks = []
    processes = []
    n_users = interactions.n_users
    mp_batch = int(np.ceil(n_users / num_workers))

    queue = mp.Queue()
    rows = np.arange(n_users)
    np.random.shuffle(rows)
    for rank in range(num_workers):
        start = rank * mp_batch
        end = np.min((start + mp_batch, n_users))
        p = mp.Process(target=batch_patk,
                       args=(queue, rows[start:end], interactions, model),
                       kwargs={'k': k})
        p.start()
        processes.append(p)

    while True:
        is_alive = False
        for p in processes:
            if p.is_alive():
                is_alive = True
                break
        if not is_alive and queue.empty():
            break

        while not queue.empty():
            patks.append(queue.get())

    queue.close()
    for p in processes:
        p.join()
    return np.mean(patks)


def batch_patk(queue, rows, interactions, model, k=5):
    n_items = interactions.n_items

    items = torch.arange(0, n_items).long()
    users_init = torch.ones(n_items).long()
    for row in rows:
        row = int(row)
        users = users_init.fill_(row)

        preds = model.predict(users, items)
        actuals = get_row_indices(row, interactions)

        # if len(actuals) == 0:
        #     continue

        top_k = np.argpartition(-np.squeeze(preds.data.numpy()), k)
        top_k = set(top_k[:k])

        true_pids = set(actuals)
        if true_pids:
            queue.put(len(top_k & true_pids) / float(k))


def compute_gini(x, w=None):
    x = np.asarray(x)
    if w is not None:
        w = np.asarray(w)
        sorted_indices = np.argsort(x)
        sorted_x = x[sorted_indices]
        sorted_w = w[sorted_indices]
        cumw = np.cumsum(sorted_w, dtype=float)
        cumxw = np.cumsum(sorted_x * sorted_w, dtype=float)
        return (np.sum(cumxw[1:] * cumw[:-1] - cumxw[:-1] * cumw[1:]) /
                (cumxw[-1] * cumw[-1]))
    else:
        sorted_x = np.sort(x)
        n = len(x)
        cumx = np.cumsum(sorted_x, dtype=float)
        return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n


def parity(head_tail_items,long_tail_items,top_K_list):
    popular_group_num = 0
    long_tailed_group_num = 0

    for i in top_K_list:
        if int(i) in head_tail_items:
            popular_group_num += 1
        else:
            if int(i) in long_tail_items:
                long_tailed_group_num += 1
            else:
                print('key not found in both head_tail and long_tail group')

    return popular_group_num, long_tailed_group_num


def model_disparity(popular_group_num, long_tailed_group_num, g_0_num, g_1_num, alpha, lambda_tradeoff):
    phi_DP = g_1_num * popular_group_num - g_0_num * long_tailed_group_num # Ψ_𝐷𝑃 = |G1| · Exposure (G0|R𝐾 ) − |G0| · Exposure (G1|R𝐾 )
    phi_EK = popular_group_num - alpha * long_tailed_group_num # Ψ_𝐸𝐾 = Exposure (G0|R𝐾 ) − 𝛼 · Exposure (G1|R𝐾 )

    return phi_DP + lambda_tradeoff * phi_EK



