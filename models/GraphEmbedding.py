import numpy as np
# from data import MovieLensDataset
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import warnings
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.functional import edge_softmax
from utils.dataset import DoubanMovie, Yelp, LastFM

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
warnings.filterwarnings('ignore')

torch.manual_seed(0)

class HINSageLayer(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 node_dict,
                 edge_dict,
                 n_heads,
                 dropout = 0.2,
                 use_norm = False):
        super(HINSageLayer, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.num_types = len(node_dict)
        self.num_relations = len(edge_dict)
        self.total_rel = self.num_types * self.num_relations * self.num_types
        self.n_heads = n_heads
        self.d_k = out_dim // n_heads
        self.sqrt_dk = math.sqrt(self.d_k)
        self.att = None

        self.k_linears = nn.ModuleList()
        self.q_linears = nn.ModuleList()
        self.v_linears = nn.ModuleList()
        self.a_linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.use_norm = use_norm

        for t in range(self.num_types):
            self.k_linears.append(nn.Linear(in_dim,   out_dim))
            self.q_linears.append(nn.Linear(in_dim,   out_dim))
            self.v_linears.append(nn.Linear(in_dim,   out_dim))
            self.a_linears.append(nn.Linear(out_dim,  out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))

        self.relation_pri = nn.Parameter(torch.ones(self.num_relations, self.n_heads))
        self.relation_att = nn.Parameter(torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg = nn.Parameter(torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
        self.skip = nn.Parameter(torch.ones(self.num_types))
        self.drop = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    def forward(self, G, h):
        with G.local_scope():
            node_dict, edge_dict = self.node_dict, self.edge_dict
            for srctype, etype, dsttype in G.canonical_etypes:
                sub_graph = G[srctype, etype, dsttype]

                k_linear = self.k_linears[node_dict[srctype]]
                v_linear = self.v_linears[node_dict[srctype]]
                q_linear = self.q_linears[node_dict[dsttype]]

                k = k_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                v = v_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                q = q_linear(h[dsttype]).view(-1, self.n_heads, self.d_k)

                e_id = self.edge_dict[etype]

                relation_att = self.relation_att[e_id]
                relation_pri = self.relation_pri[e_id]
                relation_msg = self.relation_msg[e_id]

                k = torch.einsum("bij,ijk->bik", k, relation_att)
                v = torch.einsum("bij,ijk->bik", v, relation_msg)

                sub_graph.srcdata['k'] = k
                sub_graph.dstdata['q'] = q
                sub_graph.srcdata['v_%d' % e_id] = v

                sub_graph.apply_edges(fn.v_dot_u('q', 'k', 't'))
                attn_score = sub_graph.edata.pop('t').sum(-1) * relation_pri / self.sqrt_dk
                attn_score = edge_softmax(sub_graph, attn_score, norm_by='dst')

                sub_graph.edata['t'] = attn_score.unsqueeze(-1)

            G.multi_update_all({etype : (fn.u_mul_e('v_%d' % e_id, 't', 'm'), fn.sum('m', 't')) \
                                for etype, e_id in edge_dict.items()}, cross_reducer = 'mean')

            new_h = {}
            for ntype in G.ntypes:
                '''
                    Step 3: Target-specific Aggregation
                    x = norm( W[node_type] * gelu( Agg(x) ) + x )
                '''
                n_id = node_dict[ntype]
                alpha = torch.sigmoid(self.skip[n_id])
                t = G.nodes[ntype].data['t'].view(-1, self.out_dim)
                trans_out = self.drop(self.a_linears[n_id](t))
                trans_out = trans_out * alpha + h[ntype] * (1-alpha)
                if self.use_norm:
                    new_h[ntype] = self.norms[n_id](trans_out)
                else:
                    new_h[ntype] = trans_out
            return new_h

class GraphEmbeddingModule(nn.Module):
    def __init__(self, G, node_dict, edge_dict, n_inp, n_hid, n_out, n_layers, n_heads, use_norm = True):
        super(GraphEmbeddingModule, self).__init__()
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.gcs = nn.ModuleList()
        self.n_inp = n_inp
        self.n_hid = n_hid
        self.n_out = n_out
        self.n_layers = n_layers
        self.adapt_ws  = nn.ModuleList()
        for t in range(len(node_dict)):
            self.adapt_ws.append(nn.Linear(n_inp,   n_hid))
        for _ in range(n_layers):
            self.gcs.append(HINSageLayer(n_hid, n_hid, node_dict, edge_dict, n_heads, use_norm = use_norm))
        self.out = nn.Linear(n_hid, n_out)

    def forward(self, G, out_key):
        h = {}
        for ntype in G.ntypes:
            n_id = self.node_dict[ntype]
            h[ntype] = F.gelu(self.adapt_ws[n_id](G.nodes[ntype].data['inp']))
        for i in range(self.n_layers):
            h = self.gcs[i](G, h)
        return self.out(h[out_key])



class GraphEmbedding:
    def __init__(self, args, device="cuda"):
        self.args = args

        if self.args.dataset == "Douban Movie":
            dataset = DoubanMovie(args.dataset_path)
        elif self.args.dataset == 'Yelp':
            dataset = Yelp(args.dataset_path)
        elif self.args.dataset == 'LastFM':
            dataset = LastFM(args.dataset_path)
        else:
            raise NotImplementedError("{} not supported.".format(self.args.dataset))

        self.G = dataset.G
        self.labels = dataset.labels
        self.train_idx = dataset.train_idx
        self.val_idx = dataset.val_idx
        self.test_idx = dataset.test_idx
        self.train_size = dataset.train_size
        self.val_size = dataset.val_size
        self.test_size = dataset.test_size

        print('Successfully Bulid Hetergenous Graph for {}'.format(self.args.dataset))
        print('Training/validation/test size', self.train_size, self.val_size, self.test_size)
        print(self.G)

        self.emb = self.Training()

    def forward(self, model, G, device="cuda"):
        best_val_acc = torch.tensor(0)
        best_test_acc = torch.tensor(0)
        train_step = torch.tensor(0)
        for epoch in np.arange(self.args.n_epoch) + 1:
            model.train()
            train_step += 1
            for train_idx in tqdm(self.train_loader):
                # The loss is computed only for labeled nodes.
                logits = model(G, 'user')
                loss = F.cross_entropy(logits[train_idx], self.labels[train_idx].to(device))
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clip)
                self.optimizer.step()
                self.scheduler.step(train_step)
            if epoch % 5 == 0:
                model.eval()
                logits = model(G, 'user')
                pred = logits.argmax(1).cpu()
                train_acc = (pred[train_idx] == self.labels[train_idx]).float().mean()
                val_acc = (pred[self.val_idx] == self.labels[self.val_idx]).float().mean()
                test_acc = (pred[self.test_idx] == self.labels[self.test_idx]).float().mean()
                if best_val_acc < val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc
                print(
                    'Epoch: %d LR: %.5f Loss %.4f, Train Acc %.4f, Val Acc %.4f (Best %.4f), Test Acc %.4f (Best %.4f)' % (
                        epoch,
                        self.optimizer.param_groups[0]['lr'],
                        loss.item(),
                        train_acc.item(),
                        val_acc.item(),
                        best_val_acc.item(),
                        test_acc.item(),
                        best_test_acc.item(),
                    ))

    def Training(self, device="cuda"):
        self.train_IDX = Batchwise(self.train_idx)
        self.train_loader = DataLoader(
            dataset=self.train_IDX,
            batch_size=256,
            num_workers=8)

        node_dict = {}
        edge_dict = {}
        for ntype in self.G.ntypes:
            node_dict[ntype] = len(node_dict)
        for etype in self.G.etypes:
            edge_dict[etype] = len(edge_dict)
            self.G.edges[etype].data['id'] = torch.ones(self.G.number_of_edges(etype), dtype=torch.long) * edge_dict[
                etype]

            #     Random initialize input feature
        for ntype in self.G.ntypes:
            emb = nn.Parameter(torch.Tensor(self.G.number_of_nodes(ntype), self.args.n_inp), requires_grad=False)
            nn.init.xavier_uniform_(emb)
            self.G.nodes[ntype].data['inp'] = emb

        self.G = self.G.to(device)

        model = GraphEmbeddingModule(self.G,
                    node_dict, edge_dict,
                    n_inp=self.args.n_inp,
                    n_hid=self.args.n_hid,
                    n_out=self.labels.max() + 1,
                    n_layers=2,
                    n_heads=4,
                    use_norm=True).to(device)
        self.optimizer = torch.optim.AdamW(model.parameters())
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, total_steps=self.args.n_epoch,
                                                             max_lr=self.args.max_lr)
        print('Training Graph Embedding Module ')

        self.forward(model, self.G)
        print('Training Graph Embedding Module Complete')

class Batchwise(Dataset):
    def __init__(self, inp_data):
        self.inp_data = inp_data

    def __getitem__(self, index):
        outputs = self.inp_data[index]
        return outputs

    def __len__(self):
        return len(self.inp_data)