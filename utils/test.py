from sklearn.model_selection import train_test_split
import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import dgl



def load_data(filepath):
    names = "UserID::MovieID::Rating::Timestamp".split("::")
    ratings = pd.read_csv(
        os.path.join(filepath, "ratings.dat"),
        sep="::",
        names=names,
        engine="python")
    ratings["Rating"] = (ratings["Rating"] > 3).astype(int)
    ratings["Timestamp"] = (
                                   ratings["Timestamp"] - ratings["Timestamp"].min()
                           ) / float(ratings["Timestamp"].max() - ratings["Timestamp"].min())

    names = "UserID::Gender::Age::Occupation::Zip-code".split("::")
    users = pd.read_csv(
        os.path.join(filepath, "users.dat"),
        sep="::",
        names=names,
        engine="python")
    for i in range(1, users.shape[1]):
        users.iloc[:, i] = pd.factorize(users.iloc[:, i])[0]

    names = "MovieID::Title::Genres".split("::")
    Genres = [
        "Action", "Adventure", "Animation", "Children's", "Comedy",
        "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
        "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War",
        "Western"
    ]
    movies = pd.read_csv(
        os.path.join(filepath, "movies.dat"),
        sep="::",
        names=names,
        engine="python")
    movies["Year"] = movies["Title"].apply(lambda x: x[-5:-1])
    for genre in Genres:
        movies[genre] = movies["Genres"].apply(lambda x: genre in x)
    movies.iloc[:, 3] = pd.factorize(movies.iloc[:, 3])[0]
    movies.iloc[:, 4:] = movies.iloc[:, 4:].astype(float)
    movies = movies.loc[:, ["MovieID", "Year"] + Genres]
    movies.iloc[:, 2:] = movies.iloc[:, 2:].div(
        movies.iloc[:, 2:].sum(axis=1), axis=0)

    movie_id_map = {}
    for i in range(movies.shape[0]):
        movie_id_map[movies.loc[i, "MovieID"]] = i + 1

    movies["MovieID"] = movies["MovieID"].apply(lambda x: movie_id_map[x])
    ratings["MovieID"] = ratings["MovieID"].apply(
        lambda x: movie_id_map[x])

    # self.NUM_ITEMS = len(movies.MovieID.unique())
    # self.NUM_YEARS = len(movies.Year.unique())
    # self.NUM_GENRES = movies.shape[1] - 2
    #
    # self.NUM_USERS = len(users.UserID.unique())
    # self.NUM_OCCUPS = len(users.Occupation.unique())
    # self.NUM_AGES = len(users.Age.unique())
    # self.NUM_ZIPS = len(users["Zip-code"].unique())

    return ratings, users, movies
def rating_data(path):

        def _discount_and_norm_rewards(rewards, gamma=.9):
            rewards = list(rewards)
            discounted_episode_rewards = np.zeros_like(rewards, dtype='float64')
            cumulative = 0
            for t in reversed(range(len(rewards))):
                cumulative = cumulative * gamma + rewards[t]
                discounted_episode_rewards[t] = cumulative
            # Normalize the rewards
            # discounted_episode_rewards -= np.mean(discounted_episode_rewards)
            # discounted_episode_rewards /= np.std(discounted_episode_rewards)
            return discounted_episode_rewards

        cols = ['userid','itemid','rating']
        ratings = pd.read_table(os.path.join(path,'user_movie.dat'), sep="\t", names=cols)
        # ratings.sort_values(by=['userid','timestamp'],inplace=True)
        items = list(sorted(ratings.itemid.unique()))
        key_to_id_item = dict(zip(items,range(len(items))))
        ratings.itemid = ratings.itemid.map(key_to_id_item)
        ratings['rewards'] = ratings.groupby('userid')['rating'].transform(_discount_and_norm_rewards)

        # 切分训练集和测试集
        train_u,test_u = train_test_split(ratings.userid.unique(),test_size=0.2,random_state=1126)
        train = ratings[ratings.userid.isin(train_u)]
        test = ratings[ratings.userid.isin(test_u)]
        #
        # suffix = path[path.find('_')+1:path.find('.csv')]
        # train_path = '.train_ratings_{}.csv'.format(suffix)
        # test_path = '.test_ratings_{}.csv'.format(suffix)
        # if not os.path.exists(train_path):
        #     train.to_csv(train_path,index=None)
        #     test.to_csv(test_path,index=None)
        return ratings, train,test,ratings.itemid.nunique()

def context_data(path):
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

    return UvsU,UvsG,MvsA,MvsD,MvsT


def Hetero_graph(ratings, UvsU, UvsG, MvsA, MvsD, MvsT):
    # Build graph
    G = dgl.heterograph({
        ('user', 'rate', 'movie'): (ratings.iloc[:, 0], ratings.iloc[:, 1]),
        ('movie', 'rateBy', 'user'): (ratings.iloc[:, 1], ratings.iloc[:, 0]),

        ('user', 'has', 'friend'): (UvsU.iloc[:, 0], UvsU.iloc[:, 1]),
        ('friend', 'isFriendOf', 'user'): (UvsU.iloc[:, 1], UvsU.iloc[:, 0]),

        ('user', 'in', 'group'): (UvsG.iloc[:, 0], UvsG.iloc[:, 1]),
        ('group', 'exist', 'user'): (UvsG.iloc[:, 1], UvsG.iloc[:, 0]),

        ('movie', 'actBy', 'actor'): (MvsA.iloc[:, 0], MvsA.iloc[:, 1]),
        ('actor', 'act', 'movie'): (MvsA.iloc[:, 1], MvsA.iloc[:, 0]),

        ('movie', 'directBy', 'director'): (MvsD.iloc[:, 0], MvsD.iloc[:, 1]),
        ('director', 'direct', 'movie'): (MvsD.iloc[:, 1], MvsD.iloc[:, 0]),

        ('movie', 'isOf', 'type'): (MvsT.iloc[:, 0], MvsT.iloc[:, 1]),
        ('type', 'has', 'movie'): (MvsT.iloc[:, 1], MvsT.iloc[:, 0]),
    })

    labels = ratings
    uid = ratings['userid']
    shuffle = np.random.permutation(uid)  # 1000209
    # 80%/10\%/10\%
    # 60\%/20\%/20\%

    train_idx = torch.tensor(shuffle[0:600000]).long()
    val_idx = torch.tensor(shuffle[600000:800000]).long()
    test_idx = torch.tensor(shuffle[800000:]).long()

    return G,labels, train_idx, val_idx, test_idx, len(train_idx), len(val_idx), len(test_idx)

def simulation_reward(path):
    def _discount_and_norm_rewards(rewards, gamma=.9):
        rewards = list(rewards)
        discounted_episode_rewards = np.zeros_like(rewards, dtype='float64')
        cumulative = 0
        for t in reversed(range(len(rewards))):
            cumulative = cumulative * gamma + rewards[t]
            discounted_episode_rewards[t] = cumulative
        # Normalize the rewards
        # discounted_episode_rewards -= np.mean(discounted_episode_rewards)
        # discounted_episode_rewards /= np.std(discounted_episode_rewards)
        return discounted_episode_rewards

    cols = ['userid', 'itemid', 'rating']
    ratings = pd.read_table(os.path.join(path, 'user_movie.dat'), sep="\t", names=cols)
    items = list(sorted(ratings.itemid.unique()))
    key_to_id_item = dict(zip(items, range(len(items))))
    ratings.itemid = ratings.itemid.map(key_to_id_item)
    ratings['rewards'] = ratings.groupby('userid')['rating'].transform(_discount_and_norm_rewards)
    print(ratings)
    return ratings

ratings=simulation_reward('../Data/Douban Movie')
# print(ratings)
#                                                                                                items)
# dataset_path = "../data/"
# ratings, users, items = load_data(dataset_path)
# G, labels, train_idx, val_idx, test_idx, train_size, val_size, test_size = bulid_ML_graph(ratings, users,
#                                                                                                items)