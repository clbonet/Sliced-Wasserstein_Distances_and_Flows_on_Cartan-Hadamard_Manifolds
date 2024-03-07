import torch
import argparse

import numpy as np

from load_datasets import get_BBC, get_movies, get_goodreads
from wmd import compute_matrix_w, compute_matrix_sw

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("--loss", type=str, default="sw", help="Which loss to use, sw or w2")
parser.add_argument("--dataset", type=str, default="BBC", help="Which dataset to use")
parser.add_argument("--n_projs", type=int, default=500, help="Number of projections")
parser.add_argument("--ntry", type=int, default=5, help="Number of try")
parser.add_argument("--d", type=int, default=30, help="Dimension projection")
args = parser.parse_args()
    

if __name__ == "__main__":
    
    print(device, args.loss, flush=True)
    
    n_projs = args.n_projs
    
    
    if args.dataset == "BBC":
        X, y, w, idx_train, idx_test, idx_docs, n_words = get_BBC()

    elif args.dataset == "movie":
        X, y, w, idx_train, idx_test, idx_docs, n_words = get_movies()
        
    elif args.dataset == "goodreads_genre":
        X, y, w, idx_train, idx_test, idx_docs, n_words = get_goodreads(task="genre")
        
    elif args.dataset == "goodreads_like":
        X, y, w, idx_train, idx_test, idx_docs, n_words = get_goodreads(task="likability")

#     elif args.dataset == "Twitter":
#         mat_contents = sio.loadmat("./data/twitter-emd_tr_te_split.mat")
        
#         X = mat_contents["X"][0]
#         w = mat_contents["BOW_X"][0]


    str_d = "_d" + str(args.d)
    str_loss = "_" + args.loss
    str_dataset = "_" + args.dataset
    
    if args.loss == "sw":
        str_projs = "_projs" + str(n_projs)
    else:
        str_projs = ""
               
               
    n_try = args.ntry
    
    for k in range(n_try):        
        if args.loss == "w2":
            dist_mat = compute_matrix_w(X, w, torch.eye(300, device=device, dtype=torch.float64), torch.ones(n_words, device=device), idx_docs, device=device)
        elif args.loss == "sw":
            dist_mat = compute_matrix_sw(X, w, torch.eye(300, device=device, dtype=torch.float64), torch.ones(n_words, device=device), idx_docs, n_projs, device=device)


        np.savetxt("./results"+str_dataset+"/d_wmd"+str_loss+str_dataset+str_projs+"_k"+str(k), dist_mat)