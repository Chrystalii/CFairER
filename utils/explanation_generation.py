import tensorflow.compat.v1 as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.disable_v2_behavior()
from utils.parser import parse_args
from utils.dataset import DoubanMovie
from utils.metrics import *
from models.Reinforce_CFE import CounterFE
from models.recommender import MFModule, BasePipeline
import sys
from models.GraphEmbedding import GraphEmbedding


"""initialize args and dataset"""
args_config = parse_args()
print(args_config)
device = args_config.device
print('Using dataset: {}'.format(args_config.dataset))

"""load data"""
if args_config.dataset == 'Douban Movie':
    data = DoubanMovie(args_config.dataset_path)

else:
    raise NotImplementedError("{} not supported.".format(args_config.dataset))

train, test, n_attributes = data.train, data.test, data.n_attributes

try:
    num_sampled = int(sys.argv[1])
except:
    num_sampled = 50

MF_train, MF_test = data.train_MF, data.test_MF

recommender = BasePipeline(MF_train, test=MF_test, model=MFModule,
                           n_factors=args_config.n_factors, batch_size=args_config.MF_batch_size,
                           dropout_p=args_config.dropout_p,
                           lr=args_config.MF_lr, weight_decay=args_config.weight_decay,
                           optimizer=torch.optim.Adam, n_epochs=args_config.MF_epoch,
                           verbose=True, random_seed=2017)

rec_model = recommender.model
rec_model.load_state_dict(torch.load(os.path.join(args_config.model_out, './recommender.pt')))

# ---------------------------Graph Representation module training---------------------
context_emb = GraphEmbedding(args_config)

user_embedding = context_emb.G.nodes['user'].data['inp']
item_embedding = context_emb.G.nodes['movie'].data['inp']
friend_embedding = context_emb.G.nodes['friend'].data['inp']
group_embedding = context_emb.G.nodes['group'].data['inp']
actor_embedding = context_emb.G.nodes['actor'].data['inp']
director_embedding = context_emb.G.nodes['director'].data['inp']
type_embedding = context_emb.G.nodes['type'].data['inp']

attribute_embeddings = torch.cat((friend_embedding, group_embedding,actor_embedding, director_embedding, type_embedding), 0)  #torch.Size([13845, 128]) correct, since attributes index start with 0 to 13844


with tf.Session() as sess:

    model = CounterFE(sess, args_config, item_count=n_attributes, num_sampled=num_sampled, recommender=rec_model, attribute_embeddings = attribute_embeddings,data_arg=data)

    restore_model = model.restore_model()

    pred_prob = model.predict_att(data.users + data.items)

    print(pred_prob.shape)


# with tf.Session() as sess:
#   saver = tf.train.import_meta_graph('checkout/model_rnn_topK/reinforce_CFE-473.meta')
#   saver.restore(sess, tf.train.latest_checkpoint('checkout/model_rnn_topK/'))
#
#   graph = tf.get_default_graph()
#
#   print(data.test.shape) #(639540, 3)
#   print(data.train.shape)

#
#   print([n.name for n in tf.get_default_graph().as_graph_def().node])

  # print(graph.as_graph_def())
  # fc7 = graph.get_tensor_by_name('fc7:0')