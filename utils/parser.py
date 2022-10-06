import argparse
import math

def parse_args():
    parser = argparse.ArgumentParser(description="Run Counterfactual Explainable Fairness")
    parser.add_argument("--device", default="cuda")

    # ------------------------- Graph Embedding--------------------------------------------
    parser.add_argument('--n_epoch', type=int, default=20,help="graph embedding epoch") #20
    parser.add_argument('--n_hid', type=int, default=256,help="hidden size")
    parser.add_argument('--n_inp', type=int, default=128,help="graph embedding size")
    parser.add_argument('--clip', type=int, default=1.0,help="num of graph clip")
    parser.add_argument('--max_lr', type=float, default=1e-3,help="graph learning rate")

    # ------------------------- Recommender Training--------------------------------------------
    parser.add_argument('--train_recommender', type=bool, default=True)
    parser.add_argument('--n_factors', type=int, default=128,help='Number of latent factors (or embeddings)')
    # assert
    parser.add_argument('--MF_batch_size', type=int, default=1024)
    parser.add_argument('--dropout_p', type=float, default=0, help='dropout probability to prevent over-fitting') #0.02
    parser.add_argument('--MF_lr', type=float, default=0.02)
    parser.add_argument('--weight_decay',type=float, default=0.1)
    parser.add_argument('--MF_epoch',type=int, default=50) #50

    # ------------------------- Reinforece CFE Framework--------------------------------------------
    parser.add_argument('--ek_alpha', type=float, default=0.9) #alpha
    parser.add_argument('--lambda_tradeoff', type=float, default=1) #lambda
    parser.add_argument('--count_threshold', type=float,default=15) #epsilon
    parser.add_argument("--embedding_size", type=int, default=128, help="num of embedding size")
    parser.add_argument("--rnn_size", type=int, default=128, help="num of RNN size")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--epochs", type=int, default=20, help="training epochs") #1000 #20
    parser.add_argument("--hidden_size", type=int, default=1024, help="hidden size")
    parser.add_argument("--topK", type=int, default=10, help="Tok K number")
    parser.add_argument(
        "--Ks", nargs="?", default="[20, 40, 60, 80, 100]", help="evaluate K list"
    )
    parser.add_argument('--erasure', default=10, help='erasure evaluation rate')
    parser.add_argument('--erasure_batch', default=1024, help='erasure evaluation rate')
    parser.add_argument("--weight_capping_c", type=float, default=math.e**3, help="weight capping c value")
    parser.add_argument("--gamma", type=float, default=0.95, help="discounted factor gamma")
    parser.add_argument("--kl_targ", type=float, default=0.02, help="kl targ")
    parser.add_argument("--is_train", type=bool, default=True, help="Train or load")

    # -------------------------------------File Args--------------------------------------------
    parser.add_argument('--use_pretrain_model', type=bool, default=False, help="use pretrained CFE?")
    parser.add_argument('--dataset', default="LastFM", help="support dataset: Douban Movie, Yelp, LastFM") # 3 Name need be modified
    parser.add_argument("--dataset_path", type=str, default='./Data/LastFM', help="./Data/Douban Movie; ./Data/Yelp; ./Data/LastFM")
    parser.add_argument("--model_name", type=str, default='reinforce_CFE', help="model name")
    parser.add_argument('--model_out',type=str, default='./save_model')
    parser.add_argument("--log_out", type=str, default='./logs/', help="output directory for log")
    parser.add_argument("--checkout", type=str, default='./checkout/LastFM', help="checkout dictionary")
    parser.add_argument("--figure_path", type=str, default='./Training_Figures/', help="figure path")
    parser.add_argument('--explanation_path', type=str, default='./explanation/',help='Explanation path')
    parser.add_argument('--embedding_path', type=str, default='./embeddings/')


    return parser.parse_args()