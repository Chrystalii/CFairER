import tensorflow.compat.v1 as tf
from utils.parser import parse_args
import time
import os
from utils.dataset import DoubanMovie,Yelp, LastFM
from models.Reinforce_CFE import CounterFE
import sys
from models.GraphEmbedding import GraphEmbedding
from models.recommender import MFModule, BasePipeline
from utils.metrics import *
import pandas as pd
from datetime import datetime


class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

if __name__ == '__main__':

    """initialize args and dataset"""
    args_config = parse_args()
    print(args_config)
    device = args_config.device
    print('Using dataset: {}'.format(args_config.dataset))

    """load data"""
    if args_config.dataset == 'Douban Movie':
        data = DoubanMovie(args_config.dataset_path)
    elif args_config.dataset == 'Yelp':
        data = Yelp(args_config.dataset_path)
    elif args_config.dataset == 'LastFM':
        data = LastFM(args_config.dataset_path)
    else:
        raise NotImplementedError("{} not supported.".format(args_config.dataset))

    # sys.stdout = Logger("./log/{}_GE_{}_Rec_{}_CEF_{}_emb_{}_{}.txt".format(args_config.dataset, args_config.n_epoch, args_config.MF_epoch, args_config.epochs,args_config.embedding_size, datetime.now().strftime("%d_%b_%Y")))

    # # ---------------------------Recommendation model training---------------------------

    MF_train, MF_test = data.train_MF, data.test_MF
    
    recommender = BasePipeline(MF_train, test=MF_test, model=MFModule,
                               n_factors=args_config.n_factors, batch_size=args_config.MF_batch_size, dropout_p=args_config.dropout_p,
                               lr=args_config.MF_lr, weight_decay=args_config.weight_decay,
                               optimizer=torch.optim.Adam, n_epochs=args_config.MF_epoch,
                               verbose=True, random_seed=2017)
    
    if args_config.train_recommender:
        recommender.fit()
        torch.save(recommender.model.state_dict(),os.path.join(args_config.model_out,'./recommender_{}_MFepoch_{}.pt'.format(args_config.dataset, args_config.MF_epoch)))
    
    rec_model = recommender.model
    rec_model.load_state_dict(torch.load(os.path.join(args_config.model_out,'./recommender_{}_MFepoch_{}.pt'.format(args_config.dataset, args_config.MF_epoch))))
    print('loaded recommender paras:', rec_model.eval())

    # with torch.no_grad():
    #     top_K = get_top_K_list(data.users, data.items, rec_model, k=10)

    # ---------------------------Graph Representation module training---------------------
    context_emb = GraphEmbedding(args_config)

    if args_config.dataset =='Douban Movie':
        user_embedding = context_emb.G.nodes['user'].data['inp']
        item_embedding = context_emb.G.nodes['movie'].data['inp']
        friend_embedding = context_emb.G.nodes['friend'].data['inp']
        group_embedding = context_emb.G.nodes['group'].data['inp']
        actor_embedding = context_emb.G.nodes['actor'].data['inp']
        director_embedding = context_emb.G.nodes['director'].data['inp']
        type_embedding = context_emb.G.nodes['type'].data['inp']

        print("load embedding complete! with size: ", 'user:', user_embedding.shape, 'movie:', item_embedding.shape,
              'friend:', friend_embedding.shape, 'group:', group_embedding.shape, 'actor:', actor_embedding.shape,
              'director:', director_embedding.shape, 'type:', type_embedding.shape)

        # concat embeddings to all, follows the order of [UvsU, UvsG, MvsA, MvsD, MvsT], should be consistent with attribute index in dataset.py:  attributes = pd.concat([UvsU, UvsG, MvsA, MvsD, MvsT], axis=0)

        attribute_embeddings = torch.cat((friend_embedding, group_embedding,actor_embedding, director_embedding, type_embedding), 0)  #torch.Size([13845, 128]) correct, since attributes index start with 0 to 13844

    elif args_config.dataset =='Yelp':
        user_embedding = context_emb.G.nodes['user'].data['inp']
        item_embedding = context_emb.G.nodes['business'].data['inp']
        friend_embedding = context_emb.G.nodes['friend'].data['inp']
        compliment_embedding = context_emb.G.nodes['compliment'].data['inp']
        category_embedding = context_emb.G.nodes['category'].data['inp']
        city_embedding = context_emb.G.nodes['city'].data['inp']

        print("load embedding complete! with size: ", 'user:', user_embedding.shape, 'business:', item_embedding.shape,
              'friend:', friend_embedding.shape, 'compliment:', compliment_embedding.shape, 'category:', category_embedding.shape,
              'city:', city_embedding.shape)

        attribute_embeddings = torch.cat((friend_embedding, compliment_embedding,category_embedding, city_embedding), 0) # consistent with pd.concat([UvsU, UvsC, BvsCa, BvsCi], axis=0)
    elif args_config.dataset == 'LastFM':
        user_embedding = context_emb.G.nodes['user'].data['inp']
        item_embedding = context_emb.G.nodes['artist'].data['inp']
        friend_embedding = context_emb.G.nodes['friend'].data['inp']
        similar_artist_embedding = context_emb.G.nodes['similar_artist'].data['inp']
        tag_embedding = context_emb.G.nodes['tag'].data['inp']

        print("load embedding complete! with size: ", 'user:', user_embedding.shape, 'artist:', item_embedding.shape,
              'friend:', friend_embedding.shape, 'similar_artist:', similar_artist_embedding.shape, 'tag:',
              tag_embedding.shape)

        attribute_embeddings = torch.cat((friend_embedding, similar_artist_embedding, tag_embedding),
                                     0)  # consistent with attributes = pd.concat([UvsU, AvsA, AvsT], axis=0)

    print('saving attribute embeddings...')
    np.save(os.path.join(args_config.embedding_path,
                         '{}_attribute_emb_{}_epoch_{}.npy'.format(args_config.dataset, args_config.n_inp,
                                                                   args_config.n_epoch)),attribute_embeddings.cpu().detach().numpy())

    # ---------------------------CFE Reinforcement Learning-------------------------------------------
    train, test, n_attributes = data.train, data.test, data.n_attributes
    print('test node:{}'.format(test.node_id.nunique()), 'test attribute:{}'.format(test.attribute_id.nunique()))
    t1 = time.time()
    
    try:
        num_sampled = int(sys.argv[1])
    except:
        num_sampled = 50
    
    print('Start model training: {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t1))))
    with tf.Session() as sess:
        reinforce = CounterFE(sess, args_config, item_count=n_attributes, num_sampled=num_sampled, recommender=rec_model, attribute_embeddings = attribute_embeddings,data_arg=data)
        pi_loss, beta_loss = reinforce.train(train)
        # reinforce.plot_pi(pi_loss)
        # reinforce.plot_beta(beta_loss)
    
        t2 = time.time()
        print('Model training end: {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t2))))
        print('Time cost :{} m'.format((t2-t1)/60))
    
        print('Evaluating..................')
        res = reinforce.evaluate_sessions_batch(test)
        t2_ = time.time()
        print('Recall@20: {}\tMRR@20: {}'.format(res[0], res[1]))
        print('Evaluation time cost :{} m'.format((t2_-t2)/60))
    
        # Generate predictions for test.
        # print('Predicting counterfactual attributes from trained model.')
        # if args_config.use_pretrain_model:
        #     print('using pretrained CEF')
        #     pred_prob = predict_att_with_pretrain(data.users + data.items)
        # else:
    
        pred_prob = reinforce.predict_counterfactual_att(data.users + data.items)
        # print(pred_prob, pred_prob.shape) # [26044 rows x 13845 columns]  (node size, attribute size)
    
        top_n = 10
        top_attributes = pd.DataFrame({n: pred_prob.T[col].nlargest(top_n).index.tolist() for n, col in enumerate(pred_prob.T)}).T
        # print(top_attributes) # [26044 rows x 10 columns]
    
        top_attributes.to_csv(os.path.join(args_config.explanation_path,'counterfactual_attribute_{}_{}.csv'.format(args_config.dataset,datetime.now().strftime("%d_%b_%Y"))), index=False)
    
    





