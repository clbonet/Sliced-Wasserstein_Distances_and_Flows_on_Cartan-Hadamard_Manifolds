import torch
import argparse

import numpy as np

from load_datasets import get_BBC, get_movies, get_goodreads
from nca_wcd import train_nca_wcd


device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="BBC", help="Which dataset to use")
parser.add_argument("--n_epochs", type=int, default=10000, help="Number of epochs")
parser.add_argument("--d", type=int, default=30, help="Dimension projection")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
args = parser.parse_args()
    

if __name__ == "__main__":
    print(device, flush=True)    
    
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
    str_dataset = "_" + args.dataset
    
    for i in range(len(idx_train)):            
        A, L = train_nca_wcd(X, y, w, idx_train[i], device=device, d=args.d, n_epochs=args.n_epochs, lr=args.lr)
        np.savetxt("./results"+str_dataset+"/A_wcd"+str_d+str_dataset+"_i"+str(i), A.detach().cpu().numpy())
