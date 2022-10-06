from sklearn.model_selection import train_test_split
import pandas as pd
import os
from torch.utils.data import Dataset
import dgl
import scipy.sparse as sp
import matplotlib.pyplot as plt
from utils.metrics import *
from utils.func import interaction_dict



class DoubanMovie(Dataset):
    def __init__(self, path):
        # basic usage
        self.ratings, self.interaction_sparse,self.filtered_n_users, self.filtered_n_items, self.train_MF, self.test_MF, self.n_users, self.n_items, self.users,self.items = self.rating_data(path)
        self.interaction_dict = self._interactions(self.interaction_sparse)

        # graph embedding usage
        self.UvsU, \
        self.UvsG, \
        self.MvsA, \
        self.MvsD, \
        self.MvsT, self.users_reindex, self.items_reindex = self.context_data(path)
        self.G, \
        self.labels, \
        self.train_idx, \
        self.val_idx, \
        self.test_idx, \
        self.train_size, \
        self.val_size, \
        self.test_size \
            = self.Hetero_graph()

        # Reinforce CFE usage
        self.train, self.test, self.n_attributes, self.candidate_attributes, \
        self.n_friends,  self.n_groups,  self.n_actors,  self.n_directors, self.n_types \
            = self.candidate_attribute(self.UvsU, self.UvsG, self.MvsA, self.MvsD, self.MvsT)

        self.head_tail_items, self.long_tail_items = self.popular_long_tail_groups_split(path)

    def rating_data(self, path):
        cols = ['userid', 'itemid', 'rating']
        ratings = pd.read_table(os.path.join(path, 'user_movie.dat'), sep="\t", names=cols)
        items = list(sorted(ratings.itemid.unique()))
        users = list(sorted(ratings.userid.unique()))
        key_to_id_item = dict(zip(items, range(len(items))))
        key_to_id_user = dict(zip(users, range(len(users))))
        ratings.itemid = ratings.itemid.map(key_to_id_item)
        ratings.userid = ratings.userid.map(key_to_id_user)

        n_users = ratings.userid.unique().shape[0]
        n_items = ratings.itemid.unique().shape[0]

        interactions = np.zeros((n_users, n_items))
        for row in ratings.itertuples():
            interactions[row[1] - 1, row[2] - 1] = row[3]

        interactions = (interactions >= 4).astype(np.float32) # convert interactions to implicit data)
        interaction_sparse = sp.coo_matrix(interactions)

        train_MF, test_MF = train_test_split_MF(interactions)
        train_MF = sp.coo_matrix(train_MF)
        test_MF = sp.coo_matrix(test_MF)

        return ratings, interaction_sparse,interaction_sparse.row.max()+1,interaction_sparse.col.max()+1,train_MF, test_MF,  n_users,n_items,list(sorted(ratings.userid.unique())), list(sorted(ratings.itemid.unique()))

    @staticmethod
    def _interactions(data):
        return interaction_dict(data)

    def context_data(self, path):
        # # -------------------------------user profile---------------------------
        cols = ['userid', 'friendid', 'weight']
        UvsU = pd.read_table(
            os.path.join(path, 'user_user.dat'),
            sep="\t",
            names=cols,
        )

        cols = ['userid', 'groupid', 'weight']
        UvsG = pd.read_table(
            os.path.join(path, 'user_group.dat'),
            sep="\t",
            names=cols,
        )

        # # -------------------------------movie profile---------------------------
        cols = ['movieid', 'actorid', 'weight']
        MvsA = pd.read_table(
            os.path.join(path, 'movie_actor.dat'),
            sep="\t",
            names=cols,
        )

        cols = ['movieid', 'directorid', 'weight']
        MvsD = pd.read_table(
            os.path.join(path, 'movie_director.dat'),
            sep="\t",
            names=cols,
        )

        cols = ['movieid', 'typeid', 'weight']
        MvsT = pd.read_table(
            os.path.join(path, 'movie_type.dat'),
            sep="\t",
            names=cols,
        )

        # get Key
        friends = list(sorted(UvsU.friendid.unique()))
        key_to_id_friend = dict(zip(friends, range(len(friends))))
        UvsU.friendid = UvsU.friendid.map(key_to_id_friend)  # TODO: remap ids to adjust to attribute types

        groups = list(sorted(UvsG.groupid.unique()))
        key_to_id_group = dict(zip(groups, range(len(groups))))
        UvsG.groupid = UvsG.groupid.map(key_to_id_group)

        actors = list(sorted(MvsA.actorid.unique()))
        key_to_id_actor = dict(zip(actors, range(len(actors))))
        MvsA.actorid = MvsA.actorid.map(key_to_id_actor)

        directors = list(sorted(MvsD.directorid.unique()))
        key_to_id_director = dict(zip(directors, range(len(directors))))
        MvsD.directorid = MvsD.directorid.map(key_to_id_director)

        types = list(sorted(MvsT.typeid.unique()))
        key_to_id_type = dict(zip(types, range(len(types))))
        MvsT.typeid = MvsT.typeid.map(key_to_id_type)

        # remap attribute id from [0, #attribute) to [#last attribute index , #last attribute index + #attribute_num).
        # e.g., friend id start from 0, groupid start from max friend id
        UvsG.groupid = UvsG.groupid + UvsU.friendid.max() + 1 # start with 2294
        MvsA.actorid = MvsA.actorid + UvsG.groupid.max() + 1
        MvsD.directorid = MvsD.directorid + MvsA.actorid.max() + 1
        MvsT.typeid = MvsT.typeid + MvsD.directorid.max() + 1

        # remap movie id from [0, #attribute) to [#user id index , #user id index + #movie_num).
        MvsA.movieid = MvsA.movieid + self.ratings.userid.max() + 1
        MvsD.movieid = MvsD.movieid + self.ratings.userid.max() + 1
        MvsT.movieid = MvsT.movieid + self.ratings.userid.max() + 1

        return UvsU, UvsG, MvsA, MvsD, MvsT,list(range(0, self.ratings.userid.max())), list(range(self.ratings.userid.max()+1, self.ratings.userid.max() + 1 + self.ratings.itemid.max()))

    def Hetero_graph(self):
        # Build Heterogenous Graph

        UvsM_data=self.ratings.astype({'userid': 'category', 'itemid': 'category'})
        UvsM_user_ids = torch.LongTensor(UvsM_data['userid'].cat.codes.values)
        UvsM_movie_ids = torch.LongTensor(UvsM_data['itemid'].cat.codes.values)

        UvsU_data = self.UvsU.astype({'userid': 'category', 'friendid': 'category'})
        UvsU_user_ids = torch.LongTensor(UvsU_data['userid'].cat.codes.values)
        UvsU_friend_ids = torch.LongTensor(UvsU_data['friendid'].cat.codes.values)

        UvsG_data = self.UvsG.astype({'userid': 'category', 'groupid': 'category'})
        UvsG_user_ids = torch.LongTensor(UvsG_data['userid'].cat.codes.values)
        UvsG_group_ids = torch.LongTensor(UvsG_data['groupid'].cat.codes.values)

        MvsA_data = self.MvsA.astype({'movieid': 'category', 'actorid': 'category'})
        UvsA_user_ids = torch.LongTensor(MvsA_data['movieid'].cat.codes.values)
        UvsA_actor_ids = torch.LongTensor(MvsA_data['actorid'].cat.codes.values)

        MvsD_data = self.MvsD.astype({'movieid': 'category', 'directorid': 'category'})
        UvsD_user_ids = torch.LongTensor(MvsD_data['movieid'].cat.codes.values)
        UvsD_director_ids = torch.LongTensor(MvsD_data['directorid'].cat.codes.values)


        MvsT_data = self.MvsT.astype({'movieid': 'category', 'typeid': 'category'})
        UvsT_user_ids = torch.LongTensor(MvsT_data['movieid'].cat.codes.values)
        UvsT_type_ids = torch.LongTensor(MvsT_data['typeid'].cat.codes.values)

        G = dgl.heterograph({
            ('user', 'rate', 'movie'): (UvsM_user_ids, UvsM_movie_ids),
            ('movie', 'rateBy', 'user'): (UvsM_movie_ids, UvsM_user_ids),

            ('user', 'hasfriend', 'friend'): (UvsU_user_ids, UvsU_friend_ids),
            ('friend', 'isFriendOf', 'user'): (UvsU_friend_ids,UvsU_user_ids),

            ('user', 'in', 'group'): (UvsG_user_ids, UvsG_group_ids),
            ('group', 'exist', 'user'):  (UvsG_group_ids,UvsG_user_ids),

            ('movie', 'actBy', 'actor'): (UvsA_user_ids, UvsA_actor_ids),
            ('actor', 'act', 'movie'): (UvsA_actor_ids, UvsA_user_ids),

            ('movie', 'directBy', 'director'): (UvsD_user_ids, UvsD_director_ids),
            ('director', 'direct', 'movie'): (UvsD_director_ids,UvsD_user_ids),

            ('movie', 'isOf', 'type'): (UvsT_user_ids, UvsT_type_ids),
            ('type', 'occupy', 'movie'): (UvsT_type_ids,UvsT_user_ids),
        })

        rating_long = self.ratings.astype({'userid': 'category', 'itemid': 'category'})
        labels = torch.LongTensor(rating_long['itemid'].cat.codes.values)

        uid = self.ratings['userid']  # (1068278,)
        shuffle = np.random.permutation(uid)
        # 80%/10\%/10\%
        # 60\%/20\%/20\%

        train_idx = torch.tensor(shuffle[0:800000]).long()
        val_idx = torch.tensor(shuffle[800000:1000000]).long()
        test_idx = torch.tensor(shuffle[1000000:]).long()

        # train_idx = torch.tensor(shuffle[0:1000]).long()
        # val_idx = torch.tensor(shuffle[1000:2000]).long()
        # test_idx = torch.tensor(shuffle[3000:4000]).long()

        return G, labels, train_idx, val_idx, test_idx, len(train_idx), len(val_idx), len(test_idx)

    def candidate_attribute(self, UvsU, UvsG, MvsA, MvsD, MvsT):
        # def _discount_and_norm_rewards(rewards, gamma=.9):
        #     rewards = list(rewards)
        #     discounted_episode_rewards = np.zeros_like(rewards, dtype='float64')
        #     cumulative = 0
        #     for t in reversed(range(len(rewards))):
        #         cumulative = cumulative * gamma + rewards[t]
        #         discounted_episode_rewards[t] = cumulative
        #     # Normalize the rewards
        #     # discounted_episode_rewards -= np.mean(discounted_episode_rewards)
        #     # discounted_episode_rewards /= np.std(discounted_episode_rewards)
        #     return discounted_episode_rewards
        #
        # UvsU['rewards'] = UvsU.groupby('userid')['weight'].transform(_discount_and_norm_rewards)
        # UvsG['rewards'] = UvsG.groupby('userid')['weight'].transform(_discount_and_norm_rewards)
        # MvsA['rewards'] = MvsA.groupby('movieid')['weight'].transform(_discount_and_norm_rewards)
        # MvsD['rewards'] = MvsD.groupby('movieid')['weight'].transform(_discount_and_norm_rewards)
        # MvsT['rewards'] = MvsT.groupby('movieid')['weight'].transform(_discount_and_norm_rewards)

        UvsU = UvsU.rename(columns={'userid': 'node_id','friendid': 'attribute_id'})
        UvsG = UvsG.rename(columns={'userid': 'node_id','groupid': 'attribute_id'})

        MvsA = MvsA.rename(columns={'movieid': 'node_id','actorid': 'attribute_id'})
        MvsD = MvsD.rename(columns={'movieid': 'node_id','directorid': 'attribute_id'})
        MvsT = MvsT.rename(columns={'movieid': 'node_id','typeid': 'attribute_id'})
        attributes = pd.concat([UvsU, UvsG, MvsA, MvsD, MvsT], axis=0) #TODO: attribute concatenation, directly concate then select attribute embedding according to range(0:n_friends)

        #print(len(UvsU.attribute_id.unique()), len(UvsG.attribute_id.unique()),len(MvsA.attribute_id.unique()), len(MvsD.attribute_id.unique()), len(MvsT.attribute_id.unique()))
        #2294 2753 6311 2449 38 = 13845

        #print(attributes.shape) #(646648, 4)

        # 切分训练集和测试集
        train_u, test_u = train_test_split(attributes.node_id.unique(), test_size=0.4, random_state=1126) #test_size=0.2
        train = attributes[attributes.node_id.isin(train_u)]
        test = attributes[attributes.node_id.isin(test_u)]


        return train, test, attributes.attribute_id.nunique(), attributes, UvsU.attribute_id.nunique(), UvsG.attribute_id.nunique(),MvsA.attribute_id.nunique(), MvsD.attribute_id.nunique(), MvsT.attribute_id.nunique()

    def popular_long_tail_groups_split(self, path):

        item_pop = self.ratings.groupby(['itemid']).count().sort_values('userid', ascending=False)['userid']
        head_tail_split = int(self.n_items * 0.2) # Top 20\% items as head tail items

        head_tail_items = np.array(item_pop[:head_tail_split].index).tolist()
        long_tail_items = np.array(item_pop[head_tail_split:].index).tolist()

        print('Head Tail', compute_gini(item_pop[:head_tail_split]))
        print('Long Tail', compute_gini(item_pop[head_tail_split:]))

        plt.rcParams.update({'font.size': 8})
        plt.figure(figsize=(6, 3.5))

        plt.title(r'Long-tailed distribution of dataset')
        plt.xlabel('Items interacted by popularity')
        plt.ylabel('Number of interactions in the dataset')
        plt.plot(range(head_tail_split), item_pop.values[:head_tail_split], alpha=0.7, label=r'Popular group')
        plt.plot(range(head_tail_split, len(item_pop.index)), item_pop.values[head_tail_split:], label=r'long-tailed group')
        plt.axhline(y=item_pop.values[head_tail_split], linestyle='--', lw=1, c='grey')
        plt.axvline(x=head_tail_split, linestyle='--', lw=1, c='grey')
        plt.xlim([-25, len(item_pop.index)])
        plt.ylim([-25, item_pop.values[0]])
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(path,'long_tail_distribution.png'))

        return head_tail_items, long_tail_items

class Yelp(Dataset):
    def __init__(self, path):
        self.ratings,  self.interaction_sparse,self.filtered_n_users, self.filtered_n_items, self.train_MF, self.test_MF, self.n_users, self.n_items, self.users,self.items = self.rating_data(path)
        self.interaction_dict = self._interactions(self.interaction_sparse)

        self.UvsU, self.UvsC, self.BvsCa, self.BvsCi, self.users_reindex, self.items_reindex = self.context_data(path)
        self.G, \
        self.labels, \
        self.train_idx, \
        self.val_idx, \
        self.test_idx, \
        self.train_size, \
        self.val_size, \
        self.test_size \
            = self.Hetero_graph()

        self.train, self.test, self.n_attributes, self.candidate_attributes, \
        self.n_friends,  self.n_compliments,  self.n_categories,  self.n_cities = self.candidate_attribute(self.UvsU, self.UvsC, self.BvsCa, self.BvsCi)

        self.head_tail_items, self.long_tail_items = self.popular_long_tail_groups_split(path)

    @staticmethod
    def _interactions(data):
        return interaction_dict(data)

    def rating_data(self, path):
        cols = ['userid', 'itemid', 'rating']
        ratings = pd.read_table(os.path.join(path, 'user_business.dat'), sep="\t", names=cols)
        items = list(sorted(ratings.itemid.unique()))
        users = list(sorted(ratings.userid.unique()))
        key_to_id_item = dict(zip(items, range(len(items))))
        key_to_id_user = dict(zip(users, range(len(users))))
        ratings.itemid = ratings.itemid.map(key_to_id_item)
        ratings.userid = ratings.userid.map(key_to_id_user)

        n_users = ratings.userid.unique().shape[0]
        n_items = ratings.itemid.unique().shape[0]

        interactions = np.zeros((n_users, n_items))
        for row in ratings.itertuples():
            interactions[row[1] - 1, row[2] - 1] = row[3]


        interactions = (interactions >= 4).astype(np.float32) # convert interactions to implicit data)
        interaction_sparse = sp.coo_matrix(interactions)

        train_MF, test_MF = train_test_split_MF(interactions)
        train_MF = sp.coo_matrix(train_MF)
        test_MF = sp.coo_matrix(test_MF)

        return ratings, interaction_sparse,interaction_sparse.row.max()+1,interaction_sparse.col.max()+1,train_MF, test_MF,  n_users,n_items,list(sorted(ratings.userid.unique())), list(sorted(ratings.itemid.unique()))

    def context_data(self, path):
        # # -------------------------------user profile---------------------------
        cols = ['userid', 'friendid', 'weight']
        UvsU = pd.read_table(
            os.path.join(path, 'user_user.dat'),
            sep="\t",
            names=cols,
        )

        cols = ['userid', 'complimentid', 'weight']
        UvsC = pd.read_table(
            os.path.join(path, 'user_compliment.dat'),
            sep="\t",
            names=cols,
        )

        # # -------------------------------business profile---------------------------
        cols = ['businessid', 'categoryid', 'weight']
        BvsCa = pd.read_table(
            os.path.join(path, 'business_category.dat'),
            sep="\t",
            names=cols,
        )

        cols = ['businessid', 'cityid', 'weight']
        BvsCi = pd.read_table(
            os.path.join(path, 'business_city.dat'),
            sep="\t",
            names=cols,
        )

        friends = list(sorted(UvsU.friendid.unique()))
        key_to_id_friend = dict(zip(friends, range(len(friends))))
        UvsU.friendid = UvsU.friendid.map(key_to_id_friend)  # TODO: remap ids to adjust to attribute types

        compliments = list(sorted(UvsC.complimentid.unique()))
        key_to_id_group = dict(zip(compliments, range(len(compliments))))
        UvsC.complimentid = UvsC.complimentid.map(key_to_id_group)

        categories = list(sorted(BvsCa.categoryid.unique()))
        key_to_id_actor = dict(zip(categories, range(len(categories))))
        BvsCa.categoryid = BvsCa.categoryid.map(key_to_id_actor)

        cities = list(sorted(BvsCi.cityid.unique()))
        key_to_id_director = dict(zip(cities, range(len(cities))))
        BvsCi.cityid = BvsCi.cityid.map(key_to_id_director)

        UvsC.complimentid = UvsC.complimentid + UvsU.friendid.max() + 1 # start with 2294
        BvsCa.categoryid = BvsCa.categoryid + UvsC.complimentid.max() + 1
        BvsCi.cityid = BvsCi.cityid + BvsCa.categoryid.max() + 1

        # remap movie id from [0, #attribute) to [#user id index , #user id index + #movie_num).
        BvsCa.businessid = BvsCa.businessid + self.ratings.userid.max() + 1
        BvsCi.businessid = BvsCi.businessid + self.ratings.userid.max() + 1

        return UvsU, UvsC, BvsCa, BvsCi,list(range(0, self.ratings.userid.max())), list(range(self.ratings.userid.max()+1, self.ratings.userid.max() + 1 + self.ratings.itemid.max()))

    def Hetero_graph(self):

        UvsB_data=self.ratings.astype({'userid': 'category', 'itemid': 'category'})
        UvsB_user_ids = torch.LongTensor(UvsB_data['userid'].cat.codes.values)
        UvsB_business_ids = torch.LongTensor(UvsB_data['itemid'].cat.codes.values)

        UvsU_data = self.UvsU.astype({'userid': 'category', 'friendid': 'category'})
        UvsU_user_ids = torch.LongTensor(UvsU_data['userid'].cat.codes.values)
        UvsU_friend_ids = torch.LongTensor(UvsU_data['friendid'].cat.codes.values)

        UvsC_data = self.UvsC.astype({'userid': 'category', 'complimentid': 'category'})
        UvsC_user_ids = torch.LongTensor(UvsC_data['userid'].cat.codes.values)
        UvsC_compliment_ids = torch.LongTensor(UvsC_data['complimentid'].cat.codes.values)

        BvsCa_data = self.BvsCa.astype({'businessid': 'category', 'categoryid': 'category'})
        BvsCa_business_ids = torch.LongTensor(BvsCa_data['businessid'].cat.codes.values)
        BvsCa_category_ids = torch.LongTensor(BvsCa_data['categoryid'].cat.codes.values)

        BvsCi_data = self.BvsCi.astype({'businessid': 'category', 'cityid': 'category'})
        BvsCi_business_ids = torch.LongTensor(BvsCi_data['businessid'].cat.codes.values)
        BvsCi_city_ids = torch.LongTensor(BvsCi_data['cityid'].cat.codes.values)


        G = dgl.heterograph({
            ('user', 'rate', 'business'): (UvsB_user_ids, UvsB_business_ids),
            ('business', 'rateBy', 'user'): (UvsB_business_ids, UvsB_user_ids),

            ('user', 'hasfriend', 'friend'): (UvsU_user_ids, UvsU_friend_ids),
            ('friend', 'isFriendOf', 'user'): (UvsU_friend_ids,UvsU_user_ids),

            ('user', 'comply', 'compliment'): (UvsC_user_ids, UvsC_compliment_ids),
            ('compliment', 'compliedBy', 'user'):  (UvsC_compliment_ids,UvsC_user_ids),

            ('business', 'isOf', 'category'): (BvsCa_business_ids, BvsCa_category_ids),
            ('category', 'contains', 'business'): (BvsCa_category_ids, BvsCa_business_ids),

            ('business', 'locatedIn', 'city'): (BvsCi_business_ids, BvsCi_city_ids),
            ('city', 'occupy', 'business'): (BvsCi_city_ids,BvsCi_business_ids),
        })

        rating_long = self.ratings.astype({'userid': 'category', 'itemid': 'category'})
        labels = torch.LongTensor(rating_long['itemid'].cat.codes.values)

        uid = self.ratings['userid']  # (1068278,)
        shuffle = np.random.permutation(uid)
        # 80%/10\%/10\%
        # 60\%/20\%/20\%
        #
        train_idx = torch.tensor(shuffle[0:800000]).long()
        val_idx = torch.tensor(shuffle[800000:1000000]).long()
        test_idx = torch.tensor(shuffle[1000000:]).long()

        # train_idx = torch.tensor(shuffle[0:1000]).long()
        # val_idx = torch.tensor(shuffle[1000:2000]).long()
        # test_idx = torch.tensor(shuffle[3000:4000]).long()

        return G, labels, train_idx, val_idx, test_idx, len(train_idx), len(val_idx), len(test_idx)

    def candidate_attribute(self, UvsU, UvsC, BvsCa, BvsCi):

        UvsU = UvsU.rename(columns={'userid': 'node_id','friendid': 'attribute_id'})
        UvsC = UvsC.rename(columns={'userid': 'node_id','complimentid': 'attribute_id'})

        BvsCa = BvsCa.rename(columns={'businessid': 'node_id','categoryid': 'attribute_id'})
        BvsCi = BvsCi.rename(columns={'businessid': 'node_id','cityid': 'attribute_id'})

        attributes = pd.concat([UvsU, UvsC, BvsCa, BvsCi], axis=0)

        # 切分训练集和测试集
        train_u, test_u = train_test_split(attributes.node_id.unique(), test_size=0.4, random_state=1126) #test_size=0.4
        train = attributes[attributes.node_id.isin(train_u)]
        test = attributes[attributes.node_id.isin(test_u)]

        return train, test, attributes.attribute_id.nunique(), attributes, UvsU.attribute_id.nunique(), UvsC.attribute_id.nunique(),BvsCa.attribute_id.nunique(), BvsCi.attribute_id.nunique()

    def popular_long_tail_groups_split(self, path):

        item_pop = self.ratings.groupby(['itemid']).count().sort_values('userid', ascending=False)['userid']
        head_tail_split = int(self.n_items * 0.2) # Top 20\% items as head tail items

        head_tail_items = np.array(item_pop[:head_tail_split].index).tolist()
        long_tail_items = np.array(item_pop[head_tail_split:].index).tolist()

        print('Head Tail', compute_gini(item_pop[:head_tail_split]))
        print('Long Tail', compute_gini(item_pop[head_tail_split:]))

        plt.rcParams.update({'font.size': 8})
        plt.figure(figsize=(6, 3.5))

        plt.title(r'Long-tailed distribution of dataset')
        plt.xlabel('Items interacted by popularity')
        plt.ylabel('Number of interactions in the dataset')
        plt.plot(range(head_tail_split), item_pop.values[:head_tail_split], alpha=0.7, label=r'Popular group')
        plt.plot(range(head_tail_split, len(item_pop.index)), item_pop.values[head_tail_split:], label=r'long-tailed group')
        plt.axhline(y=item_pop.values[head_tail_split], linestyle='--', lw=1, c='grey')
        plt.axvline(x=head_tail_split, linestyle='--', lw=1, c='grey')
        plt.xlim([-25, len(item_pop.index)])
        plt.ylim([-25, item_pop.values[0]])
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(path,'long_tail_distribution.png'))

        return head_tail_items, long_tail_items

class LastFM(Dataset):
    def __init__(self, path):
        self.ratings,self.interaction_sparse,self.filtered_n_users, self.filtered_n_items, self.train_MF, self.test_MF, self.n_users, self.n_items, self.users,self.items = self.rating_data(path)
        self.interaction_dict = self._interactions(self.interaction_sparse)

        self.UvsU, self.AvsA, self.AvsT, self.users_reindex, self.items_reindex = self.context_data(path)
        self.G, \
        self.labels, \
        self.train_idx, \
        self.val_idx, \
        self.test_idx, \
        self.train_size, \
        self.val_size, \
        self.test_size \
            = self.Hetero_graph()

        self.train, self.test, self.n_attributes, self.candidate_attributes, \
        self.n_friends, self.n_similar_artists,  self.n_tags = self.candidate_attribute(self.UvsU, self.AvsA, self.AvsT)

        self.head_tail_items, self.long_tail_items = self.popular_long_tail_groups_split(path)

    @staticmethod
    def _interactions(data):
        return interaction_dict(data)


    def rating_data(self, path):
        cols = ['userid', 'itemid', 'rating']
        ratings = pd.read_table(os.path.join(path, 'user_artist.dat'), sep="\t", names=cols)
        items = list(sorted(ratings.itemid.unique()))
        users = list(sorted(ratings.userid.unique()))
        key_to_id_item = dict(zip(items, range(len(items))))
        key_to_id_user = dict(zip(users, range(len(users))))
        ratings.itemid = ratings.itemid.map(key_to_id_item)
        ratings.userid = ratings.userid.map(key_to_id_user)

        n_users = ratings.userid.unique().shape[0]
        n_items = ratings.itemid.unique().shape[0]

        interactions = np.zeros((n_users, n_items))
        for row in ratings.itertuples():
            interactions[row[1] - 1, row[2] - 1] = row[3]

        interactions = (interactions >= 4).astype(np.float32)  # convert interactions to implicit data)
        interaction_sparse = sp.coo_matrix(interactions)

        train_MF, test_MF = train_test_split_MF(interactions)
        train_MF = sp.coo_matrix(train_MF)
        test_MF = sp.coo_matrix(test_MF)

        return ratings, interaction_sparse, interaction_sparse.row.max() + 1, interaction_sparse.col.max() + 1, train_MF, test_MF, n_users, n_items, list(
            sorted(ratings.userid.unique())), list(sorted(ratings.itemid.unique()))


    def context_data(self, path):
        # # -------------------------------user profile---------------------------
        cols = ['userid', 'friendid', 'weight']
        UvsU = pd.read_table(
            os.path.join(path, 'user_user(knn).dat'),
            sep="\t",
            names=cols,
        )

        # # -------------------------------business profile---------------------------
        cols = ['artistid', 'similar_artistid', 'weight']
        AvsA = pd.read_table(
            os.path.join(path, 'artist_artist(knn).dat'),
            sep="\t",
            names=cols,
        )

        cols = ['artistid', 'tagid', 'weight']
        AvsT = pd.read_table(
            os.path.join(path, 'artist_tag.dat'),
            sep="\t",
            names=cols,
        )

        friends = list(sorted(UvsU.friendid.unique()))
        key_to_id_friend = dict(zip(friends, range(len(friends))))
        UvsU.friendid = UvsU.friendid.map(key_to_id_friend)

        similar_artists = list(sorted(AvsA.similar_artistid.unique()))
        key_to_id_actor = dict(zip(similar_artists, range(len(similar_artists))))
        AvsA.similar_artistid = AvsA.similar_artistid.map(key_to_id_actor)

        tags = list(sorted(AvsT.tagid.unique()))
        key_to_id_director = dict(zip(tags, range(len(tags))))
        AvsT.tagid = AvsT.tagid.map(key_to_id_director)

        AvsA.similar_artistid = AvsA.similar_artistid + UvsU.friendid.max() + 1
        AvsT.tagid = AvsT.tagid + AvsA.similar_artistid.max() + 1

        AvsA.artistid = AvsA.artistid + self.ratings.userid.max() + 1
        AvsT.artistid = AvsT.artistid + self.ratings.userid.max() + 1

        return UvsU, AvsA, AvsT,list(range(0, self.ratings.userid.max())), list(range(self.ratings.userid.max()+1, self.ratings.userid.max() + 1 + self.ratings.itemid.max()))

    def Hetero_graph(self):

        UvsA_data=self.ratings.astype({'userid': 'category', 'itemid': 'category'})
        UvsA_user_ids = torch.LongTensor(UvsA_data['userid'].cat.codes.values)
        UvsA_artist_ids = torch.LongTensor(UvsA_data['itemid'].cat.codes.values)

        UvsU_data = self.UvsU.astype({'userid': 'category', 'friendid': 'category'})
        UvsU_user_ids = torch.LongTensor(UvsU_data['userid'].cat.codes.values)
        UvsU_friend_ids = torch.LongTensor(UvsU_data['friendid'].cat.codes.values)

        AvsA_data = self.AvsA.astype({'artistid': 'category', 'similar_artistid': 'category'})
        AvsA_artist_ids = torch.LongTensor(AvsA_data['artistid'].cat.codes.values)
        AvsA_similar_artist_ids = torch.LongTensor(AvsA_data['similar_artistid'].cat.codes.values)

        AvsT_data = self.AvsT.astype({'artistid': 'category', 'tagid': 'category'})
        AvsT_artist_ids = torch.LongTensor(AvsT_data['artistid'].cat.codes.values)
        AvsT_tag_ids = torch.LongTensor(AvsT_data['tagid'].cat.codes.values)


        G = dgl.heterograph({
            ('user', 'rate', 'artist'): (UvsA_user_ids, UvsA_artist_ids),
            ('artist', 'rateBy', 'user'): (UvsA_artist_ids, UvsA_user_ids),

            ('user', 'hasfriend', 'friend'): (UvsU_user_ids, UvsU_friend_ids),
            ('friend', 'isFriendOf', 'user'): (UvsU_friend_ids,UvsU_user_ids),

            ('artist', 'isSimilar', 'similar_artist'): (AvsA_artist_ids, AvsA_similar_artist_ids),
            ('similar_artist', 'similar', 'artist'):  (AvsA_similar_artist_ids,AvsA_artist_ids),

            ('artist', 'isOf', 'tag'): (AvsT_artist_ids, AvsT_tag_ids),
            ('tag', 'contains', 'artist'): (AvsT_tag_ids, AvsT_artist_ids),
        })

        rating_long = self.ratings.astype({'userid': 'category', 'itemid': 'category'})
        labels = torch.LongTensor(rating_long['itemid'].cat.codes.values)

        uid = self.ratings['userid']  # (1068278,)
        shuffle = np.random.permutation(uid)
        # 80%/10\%/10\%
        # 60\%/20\%/20\%
        #
        train_idx = torch.tensor(shuffle[0:800000]).long()
        val_idx = torch.tensor(shuffle[800000:1000000]).long()
        test_idx = torch.tensor(shuffle[1000000:]).long()

        # train_idx = torch.tensor(shuffle[0:1000]).long()
        # val_idx = torch.tensor(shuffle[1000:2000]).long()
        # test_idx = torch.tensor(shuffle[3000:4000]).long()

        return G, labels, train_idx, val_idx, test_idx, len(train_idx), len(val_idx), len(test_idx)

    def candidate_attribute(self, UvsU, AvsA, AvsT):

        UvsU = UvsU.rename(columns={'userid': 'node_id','friendid': 'attribute_id'})

        AvsA = AvsA.rename(columns={'artistid': 'node_id','similar_artistid': 'attribute_id'})
        AvsT = AvsT.rename(columns={'artistid': 'node_id','tagid': 'attribute_id'})

        attributes = pd.concat([UvsU, AvsA, AvsT], axis=0)

        # 切分训练集和测试集
        train_u, test_u = train_test_split(attributes.node_id.unique(), test_size=0.4, random_state=1126) #test_size=0.4
        train = attributes[attributes.node_id.isin(train_u)]
        test = attributes[attributes.node_id.isin(test_u)]

        return train, test, attributes.attribute_id.nunique(), attributes, UvsU.attribute_id.nunique(),AvsA.attribute_id.nunique(), AvsT.attribute_id.nunique()

    def popular_long_tail_groups_split(self, path):

        item_pop = self.ratings.groupby(['itemid']).count().sort_values('userid', ascending=False)['userid']
        head_tail_split = int(self.n_items * 0.2) # Top 20\% items as head tail items

        head_tail_items = np.array(item_pop[:head_tail_split].index).tolist()
        long_tail_items = np.array(item_pop[head_tail_split:].index).tolist()

        print('Head Tail', compute_gini(item_pop[:head_tail_split]))
        print('Long Tail', compute_gini(item_pop[head_tail_split:]))

        plt.rcParams.update({'font.size': 8})
        plt.figure(figsize=(6, 3.5))

        plt.title(r'Long-tailed distribution of dataset')
        plt.xlabel('Items interacted by popularity')
        plt.ylabel('Number of interactions in the dataset')
        plt.plot(range(head_tail_split), item_pop.values[:head_tail_split], alpha=0.7, label=r'Popular group')
        plt.plot(range(head_tail_split, len(item_pop.index)), item_pop.values[head_tail_split:], label=r'long-tailed group')
        plt.axhline(y=item_pop.values[head_tail_split], linestyle='--', lw=1, c='grey')
        plt.axvline(x=head_tail_split, linestyle='--', lw=1, c='grey')
        plt.xlim([-25, len(item_pop.index)])
        plt.ylim([-25, item_pop.values[0]])
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(path,'long_tail_distribution.png'))

        return head_tail_items, long_tail_items

def train_test_split_MF(interactions, n=10):
    test = np.zeros(interactions.shape) # (13367, 12677)
    train = interactions.copy()
    for user in range(interactions.shape[0]): # 13367
        if interactions[user, :].nonzero()[0].shape[0] > n:
            test_interactions = np.random.choice(interactions[user, :].nonzero()[0],
                                                 size=n,
                                                 replace=False)
            train[user, test_interactions] = 0.
            test[user, test_interactions] = interactions[user, test_interactions]
    # Test and training are truly disjoint

    assert(np.all((train * test) == 0))

    return train, test




