import tensorflow.compat.v1 as tf
import torch.nn.functional as F
from torch import nn
import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.disable_v2_behavior()
import pandas as pd
import os
from utils.metrics import *
cuda0 = torch.device('cuda:0')


# The implementation of target policy and logging policy with counterfactual risk minimization loss.

def concate_fusion(a_embedding, b_embedding):
    return torch.mul(a_embedding.to(cuda0), b_embedding.to(cuda0))

def get_top_K_list(node, attribute, all_attribute_embeddings, all_items, model, data_arg, args_config,k=10):

    user_max = max(data_arg.users)
    item_max = max(data_arg.items)

    if node in list(range(0, user_max)): # user node counterfactual
        ues = model.user_embeddings.weight[node].to(cuda0)
        U_cf = concate_fusion(model.user_embeddings.weight[node], all_attribute_embeddings[attribute]).to(cuda0)
        V_cf = model.item_embeddings.weight[all_items].to(cuda0)
    else:
        if node in list(range(user_max, user_max + item_max)): # item node counterfactual
            V_cf = concate_fusion(model.item_embeddings.weight[all_items],
                                  all_attribute_embeddings[attribute]).to(cuda0)
            node = random.randint(0, user_max) # select a random user for inference
            ues = model.user_embeddings.weight[node].to(cuda0)
            U_cf = model.user_embeddings.weight[node].to(cuda0)

        else:
            ues = nn.Embedding(1, args_config.n_factors, sparse=False).weight.to(cuda0)
            U_cf = nn.Embedding(1, args_config.n_factors, sparse=False).weight.to(cuda0)
            V_cf = nn.Embedding(len(all_items), args_config.n_factors, sparse=False).weight.to(cuda0)


    uis = model.item_embeddings.weight[all_items].to(cuda0)

    pred_value = (ues * uis).sum(dim=1, keepdim=True).reshape(-1)
    origin_top_k = torch.sort(pred_value,descending=True)[1][:k]
    # origin_top_k[node] = top_k_item

    invervene_preds_value = (U_cf * V_cf).sum(dim=1, keepdim=True).reshape(-1)
    intervene_top_K = torch.sort(invervene_preds_value, descending=True)[1][:k]
    # intervene_top_K[node] = invervene_top_k_item

    return origin_top_k, intervene_top_K, U_cf

def intervene_top_K(ue, uid, items, model, k=10):
    top_k = {}
    ie = model.item_embeddings.weight[items].to(cuda0)
    preds = (ue * ie).sum(dim=1, keepdim=True)
    pred_value = preds.reshape(-1)
    top_k_item = torch.sort(pred_value, descending=True)[1][:k]
    top_k[uid] = top_k_item

    return top_k

def cascade_model(p,k):
    return 1-(1-p)**k

# lambda weight
def gradient_cascade(p, k):
    return k*(1-p)**(k-1)


class CounterFE():
    def __init__(self, sess, args_config, item_count, num_sampled, recommender, attribute_embeddings, data_arg):

        self.sess = sess
        self.item_count = item_count
        self.num_sampled = num_sampled
        self.recommender = recommender
        self.attribute_embeddings = attribute_embeddings
        self.data_arg = data_arg

        self.args_config = args_config
        self.embedding_size = args_config.embedding_size
        self.rnn_size = args_config.rnn_size
        self.log_out = args_config.log_out
        self.topK = args_config.topK
        self.weight_capping_c = args_config.weight_capping_c
        self.batch_size = args_config.batch_size
        self.epochs = args_config.epochs
        self.hidden_size = args_config.hidden_size
        self.gamma = args_config.gamma
        self.model_name = args_config.model_name
        self.checkout = args_config.checkout
        self.kl_targ = args_config.kl_targ
        self.is_train = args_config.is_train
        self.figure_path = args_config.figure_path

        self.action_source = {"pi": "beta", "beta": "beta"} #由beta选择动作

        self._init_graph()

        self.saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        self.sess.run(init)

        self.log_writer = tf.summary.FileWriter(self.log_out, self.sess.graph)

        if not self.is_train:
            self.restore_model()

    def __str__(self):
        dit = self.__dict__
        show = ['item_count','embedding_size','is_train','topK','weight_capping_c','batch_size','epochs','gamma','model_name','time_step','num_sampled']
        dict = {key:val for key,val in dit.items() if key in show}
        return str(dict)


    def weight_capping(self,cof):
        return tf.minimum(cof,self.weight_capping_c)

    def choose_action(self, history):
        # Reshape observation to (num_features, 1)
        # Run forward propagation to get softmax probabilities
        prob_weights = self.sess.run(self.PI, feed_dict = {self.input: history})
        action = list(map(lambda x:np.random.choice(range(len(prob_weights.ravel())), p = prob_weights.ravel()),prob_weights))

        # Select action using a biased sample
        # this will return the index of the action we've sampled
        # action = np.random.choice(range(len(prob_weights.ravel())), p=prob_weights.ravel())

        # # exploration to allow rare data appeared
        # if random.randint(0,1000) < 1000:
        #     pass
        # else:
        #     action = random.randint(0,self.n_y-1)
        return action

    # # inference for one user
    def predict_one_session(self, history):
        state = np.zeros([1,self.rnn_size],dtype=np.float32)
        for i in history:
            alpha, state = self.sess.run([self.alpha, self.final_state], feed_dict={self.X:[i], self.state:state})

        # Reshape observation to (num_features, 1)
        # Run forward propagation to get softmax probabilities
        prob_weights = alpha
        # action = tf.arg_max(prob_weights[0])
        actions = tf.nn.top_k(prob_weights[0],self.topK)
        # tf.nn.in_top_k
        # return actions['indices'] #TopKV2(values=<tf.Tensor 'TopKV2:0' shape=(10,) dtype=float32>, indices=<tf.Tensor 'TopKV2:1' shape=(10,) dtype=int32>)

        return actions[1]

    def save_model(self,step):
        if not os.path.exists(self.checkout):
            os.makedirs(self.checkout)

        self.saver.save(self.sess,os.path.join(self.checkout,self.model_name),global_step=step,write_meta_graph=True)

    def restore_model(self):
        ckpt = tf.train.get_checkpoint_state(self.checkout)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess,os.path.join(self.checkout,ckpt_name))

    def pi_beta_sample(self):
        # 1. obtain probabilities
        # note: detach is to block gradient
        beta_probs = self.beta
        pi_probs = self.PI

        # 2. probabilities -> categorical distribution.
        beta_categorical = tf.distributions.Categorical(beta_probs)
        pi_categorical = tf.distributions.Categorical(pi_probs)

        # 3. sample the actions
        # See this issue: https://github.com/awarebayes/RecNN/issues/7
        # usually it works like:
        # pi_action = pi_categorical.sample(); beta_action = beta_categorical.sample();
        # but changing the action_source to {pi: beta, beta: beta} can be configured to be:
        # pi_action = beta_categorical.sample(); beta_action = beta_categorical.sample();
        available_actions = {
            "pi": pi_categorical.sample(),
            "beta": beta_categorical.sample(),
        }
        pi_action = available_actions[self.action_source["pi"]]
        beta_action = available_actions[self.action_source["beta"]]

        # 4. calculate stuff we need
        pi_log_prob = pi_categorical.log_prob(pi_action) # \pi_\theta (a_t|s_T)
        beta_log_prob = beta_categorical.log_prob(beta_action) # \pi_0 (a_t|s_T)

        return pi_log_prob, beta_log_prob, pi_probs

    def _init_graph(self):
        with tf.variable_scope('input'):
            self.X = tf.placeholder(tf.int32,[None],name='input')
            self.label = tf.placeholder(tf.int32,[None],name='label')
            self.discounted_episode_rewards_norm = tf.placeholder(shape=[None],name='discounted_rewards',dtype=tf.float32)
            self.state = tf.placeholder(tf.float32,[None,self.rnn_size],name='rnn_state')

        cell = tf.compat.v1.nn.rnn_cell.GRUCell(self.rnn_size)
        with tf.variable_scope('emb'):
            embedding = tf.get_variable('item_emb',[self.item_count,self.embedding_size])
            inputs = tf.nn.embedding_lookup(embedding,self.X)

        outputs,states_ = cell.__call__(inputs,self.state)  # outputs为最后一层每一时刻的输出

        self.final_state = states_

        state = outputs

        with tf.variable_scope('main_policy'):
            weights = tf.get_variable('item_emb_pi',[self.item_count,self.rnn_size])
            bias = tf.get_variable('bias',[self.item_count])
            self.pi_hat = tf.add(tf.matmul(state,tf.transpose(weights)),bias)
            self.PI = tf.nn.softmax(self.pi_hat) # PI policy (target policy)
            self.alpha = cascade_model(self.PI,self.topK)

        with tf.variable_scope('beta_policy'):
            weights_beta=tf.get_variable('item_emb_beta',[self.item_count,self.rnn_size])
            bias_beta = tf.get_variable('bias_beta',[self.item_count])
            self.beta = tf.add(tf.matmul(state,tf.transpose(weights_beta)),bias_beta)
            self.beta = tf.nn.softmax(self.beta) # β policy (logging policy)

        label = tf.reshape(self.label,[-1,1])
        with tf.variable_scope('loss'):
            pi_log_prob, beta_log_prob, pi_probs = self.pi_beta_sample()

            ce_loss_main = tf.nn.sampled_softmax_loss(
                weights,bias,label,state,self.num_sampled,num_classes=self.item_count)

            topk_correction = gradient_cascade(tf.exp(pi_log_prob),self.topK) # lambda 比值
            off_policy_correction = tf.exp(pi_log_prob)/tf.exp(beta_log_prob)
            off_policy_correction = self.weight_capping(off_policy_correction)

            self.pi_loss = tf.reduce_mean(off_policy_correction*topk_correction*self.discounted_episode_rewards_norm*ce_loss_main)
            # tf.summary.scalar('pi_loss',self.pi_loss)

            self.beta_loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(
                weights_beta,bias_beta,label,state,self.num_sampled,num_classes=self.item_count))
            # tf.summary.scalar('beta_loss',self.beta_loss)

        with tf.variable_scope('optimizer'):
            beta_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='beta_policy')
            self.train_op_pi = tf.train.AdamOptimizer(0.01).minimize(self.pi_loss)
            self.train_op_beta = tf.train.AdamOptimizer(0.01).minimize(self.beta_loss,var_list=beta_vars)

    def init(self,data):
        data.drop(['weight'],axis=1,inplace=True)
        offset_sessions = np.zeros(data.node_id.nunique()+1,dtype=np.int32)
        offset_sessions[1:] = data.groupby('node_id').size().cumsum()
        return offset_sessions

    def train(self,data):

        pi = []
        beta=[]
        counter = 0

        offset_sessions = self.init(data)
        # print(data.head(10))

        g_0, g_1 = self.data_arg.head_tail_items, self.data_arg.long_tail_items
        for epoch in range(self.epochs):
            state = np.zeros([self.batch_size,self.rnn_size],dtype=np.float32)
            session_idx_arr = np.arange(len(offset_sessions)-1)
            iters = np.arange(self.batch_size) # node id

            maxiter = iters.max()
            start = offset_sessions[session_idx_arr[iters]]
            end = offset_sessions[session_idx_arr[iters]+1]

            finished = False
            while not finished:
                minlen =(end-start).min()
                out_idx = data.attribute_id.values[start]
                for i in range(minlen-1):
                    in_idx = out_idx
                    n_idx = data.node_id.values[start + i + 1]
                    out_idx = data.attribute_id.values[start + i + 1] # ******* feed train.attribute_id
                    rewards = []
                    # calculating counterfactual reward for one batch
                    for select_attribute,node in zip(out_idx,n_idx):
                        # calculate original model disparity and counterfactual model disparity
                        origin_top_k, intervene_top_k, U_cf = get_top_K_list(node, select_attribute, self.attribute_embeddings, self.data_arg.items, self.recommender, self.data_arg, self.args_config, k=10)

                        popular_group_num, long_tailed_group_num = parity(g_0, g_1, origin_top_k)
                        origin_disparity = model_disparity(popular_group_num, long_tailed_group_num, len(g_0), len(g_1), self.args_config.ek_alpha, self.args_config.lambda_tradeoff)

                        popular_group_num, long_tailed_group_num = parity(g_0, g_1, intervene_top_k)
                        intervene_disparity = model_disparity(popular_group_num, long_tailed_group_num, len(g_0),
                                                           len(g_1), self.args_config.ek_alpha, self.args_config.lambda_tradeoff)

                        # calculate counterfactual reward
                        if origin_disparity - intervene_disparity >= self.args_config.count_threshold:
                            if node in range(0, max(self.data_arg.users)):
                                reward = 1 + F.cosine_similarity(U_cf.to(cuda0), self.recommender.user_embeddings.weight[node].to(cuda0), dim=0) # reward = 1+ cos_similarity if Δ(𝐻𝑢,𝐾 ) − Δ(𝐻𝑐𝑓 𝑢,𝐾 ) ≥ 𝜖
                            else:
                                reward = torch.tensor([1])
                        else:
                            if node in range(0, max(self.data_arg.users)):
                                reward = F.cosine_similarity(U_cf.to(cuda0), self.recommender.user_embeddings.weight[node].to(cuda0), dim=0)
                            else:
                                reward = torch.tensor([0])
                        rewards.append(reward.item())

                    rewards = np.array(rewards) # *******feed train.rewards

                    fetches =[self.final_state,self.PI,self.beta,self.pi_loss,self.beta_loss,
                              self.train_op_pi,self.train_op_beta]
                    feed_dict = {self.X:in_idx,self.label:out_idx,
                                 self.discounted_episode_rewards_norm:rewards,
                                 self.state:state}
                    state, pi_new, beta_new, pi_loss, beta_loss, _, _ = self.sess.run(fetches, feed_dict)
                    print('num_sampled:{},ite:{},epoch:{},pi loss:{:.2f},beta loss:{:.2f},finished num of user:{}/{}'.format(self.num_sampled,counter,epoch,pi_loss,beta_loss,maxiter+1,data.node_id.nunique()))

                    pi.append(pi_loss)
                    beta.append(beta_loss)
                    counter += 1

                start = start + minlen - 1
                mask = np.arange(len(iters))[(end-start) <= 1]

                for idx in mask:
                    maxiter += 1
                    if maxiter >= len(offset_sessions)-1:
                        print('epoch finished.')
                        finished=True
                        break

                    iters[idx] = maxiter
                    start[idx] = offset_sessions[session_idx_arr[maxiter]]
                    end[idx] = offset_sessions[session_idx_arr[maxiter]+1]

                if len(mask):
                    state[mask]=0

        # save model
        self.save_model(step=counter)
        return pi,beta

    def predict_att(self,data):
        self.predict_state = np.zeros([len(data), self.rnn_size], dtype=np.float32)

        in_idx = data
        print('pi_hat: ', self.pi_hat)

        fetches =[self.pi_hat,self.final_state]
        feed_dict ={self.X:in_idx,self.state:self.predict_state}
        preds, self.predict_state = self.sess.run(fetches,feed_dict) #(batch,n_items)
        preds = np.asarray(preds)

        return preds


    # 取前10个后10个的均值
    def plot_pi(self,pi_loss,num=10):
        pi_loss_ = [np.mean(pi_loss[ind-num:ind+num]) for ind ,val in enumerate(pi_loss) if ind%1000==num]
        import matplotlib.pyplot as plt
        plt.switch_backend('agg')

        plt.subplot(211)
        plt.plot(range(len(pi_loss)),pi_loss,label='pi-loss-{}'.format(self.num_sampled),color='g')
        plt.title('pi loss')

        plt.subplot(212)
        plt.plot(range(len(pi_loss_)),pi_loss_,label='pi-loss-{}'.format(self.num_sampled),color='g')
        plt.xlabel('Training Steps')
        plt.ylabel('loss')
        # plt.ylim(0,2000)
        plt.legend()
        plt.grid(True)
        # plt.show()
        plt.savefig(os.path.join(self.figure_path,'Top{}_pi_rnn_{}.jpg'.format(self.topK,self.num_sampled)))

    def plot_beta(self,beta_loss,num=10):
        # pi_loss_ = [val for ind ,val in enumerate(pi_loss) if ind%5000==0]
        # beta_loss_ = [val for ind ,val in enumerate(beta_loss) if ind%5000==0]
        beta_loss_ = [np.mean(beta_loss[ind-num:ind+num]) for ind ,val in enumerate(beta_loss) if ind%1000==num]
        import matplotlib.pyplot as plt
        plt.switch_backend('agg')

        plt.subplot(211)
        plt.plot(range(len(beta_loss)),beta_loss,label='beta-loss-{}'.format(self.num_sampled),color='r')
        plt.title('beta loss')

        plt.subplot(212)
        plt.plot(range(len(beta_loss_)),beta_loss_,label='beta-loss-{}'.format(self.num_sampled),color='r')
        plt.xlabel('Training Steps')
        plt.ylabel('loss')
        # plt.ylim(0,10)
        plt.legend()
        plt.grid(True)
        # plt.show()
        plt.savefig(os.path.join(self.figure_path,"Top{}_beta_rnn_{}.jpg".format(self.topK,self.num_sampled)))

    def predict_next_batch(self,session_ids,input_item_ids,batch): # (iters,in_idx,batch_size)
        if not self.predict:
            self.current_session = np.ones(batch)*-1
            self.predict=True

        #session_ids 即为iter, 如果此次运行的用户和上次运行的用户不一样，则说明有些用户是新加的，则需要将state置0
        session_change = np.arange(batch)[session_ids!=self.current_session] #session_ids中有些是已经结束了的,-1
        if len(session_change) > 0 :
            self.predict_state[session_change] = 0.0
            self.current_session = session_ids.copy()

        in_idx = input_item_ids
        fetches =[self.pi_hat,self.final_state]
        feed_dict ={self.X:in_idx,self.state:self.predict_state} #self.X is items/attributes
        preds,self.predict_state = self.sess.run(fetches,feed_dict) #(batch,n_items)
        preds = np.asarray(preds).T
        #print('xxxx',preds.shape) #(13845, 2048)

        return pd.DataFrame(data=preds,index=range(preds.shape[0]))

    def predict_counterfactual_att(self, node_ids):

        # init state
        self.predict_state = np.zeros([len(node_ids), self.rnn_size], dtype=np.float32)

        in_idx = node_ids
        print('pi_hat: ', self.pi_hat) # Tensor("main_policy/Add:0", shape=(?, 13845), dtype=float32) 13845 is all attribute number and 2048 is batch size i.e., node id size

        fetches =[self.pi_hat,self.final_state]
        feed_dict ={self.X:in_idx,self.state:self.predict_state}
        preds, self.predict_state = self.sess.run(fetches,feed_dict) #(batch,n_items)
        preds = np.asarray(preds)

        return pd.DataFrame(data=preds,index=range(preds.shape[0]))


    def evaluate_sessions_batch (self,test,cut_off=20,batch_size = 2048):

        self.predict = False
        offset_sessions = np.zeros(test['node_id'].nunique()+1, dtype=np.int32)
        offset_sessions[1:] = test.groupby('node_id').size().cumsum()
        evaluation_point_count = 0
        mrr,recall = 0.0,0.0
        if len(offset_sessions)-1 < batch_size:
            batch_size = len(offset_sessions)-1

        # 初始化state
        self.predict_state = np.zeros([batch_size,self.rnn_size],dtype=np.float32)

        iters = np.arange(batch_size).astype(np.int32)
        maxiter= iters.max()
        start = offset_sessions[iters]
        end = offset_sessions[iters+1]
        in_idx =np.zeros(batch_size,dtype=np.int32)
        np.random.seed(1126)
        while True:
            valid_mask = iters >= 0
            if valid_mask.sum() == 0:
                break
            start_valid = start[valid_mask]
            minlen =(end[valid_mask]-start_valid).min()
            in_idx[valid_mask] = test['attribute_id'].values[start_valid]
            for i in range(minlen-1):
                out_idx = test['attribute_id'].values[start_valid+i+1]
                preds = self.predict_next_batch(iters,in_idx,batch_size)
                preds.fillna(0, inplace=True)

                in_idx[valid_mask] = out_idx
                ranks =(preds.values.T[valid_mask].T > np.diag(preds.loc[in_idx].values)[valid_mask]).sum(axis=0) + 1
                # Top_K_rank = ranks[:self.topK]
                rank_ok = ranks < cut_off
                recall+= rank_ok.sum()
                mrr +=(1.0/ranks[rank_ok]).sum()
                evaluation_point_count += len(ranks)

            start = start+minlen-1
            mask = np.arange(len(iters))[(valid_mask)& (end-start<=1)]
            for idx in mask:
                maxiter +=1
                if maxiter>=len(offset_sessions)-1:
                    iters[idx]=-1
                else:
                    iters[idx] =maxiter
                    start[idx] =offset_sessions[maxiter]
                    end[idx] =offset_sessions[maxiter+1]

        # users = self.data_arg.users
        # preds = self.predict_one_session(users)
        # print(preds)

        return recall / evaluation_point_count, mrr / evaluation_point_count
