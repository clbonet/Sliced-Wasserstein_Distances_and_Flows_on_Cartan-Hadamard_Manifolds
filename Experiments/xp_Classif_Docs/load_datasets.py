import numpy as np
import scipy.io as sio

from sklearn.model_selection import train_test_split


def get_BBC():
    mat_contents = sio.loadmat("./data/bbcsport-emd_tr_te_split.mat")

    idx_train = mat_contents["TR"]-1
    idx_test = mat_contents["TE"]-1
    y = mat_contents["Y"][0]

    X = mat_contents["X"][0]
    w = mat_contents["BOW_X"][0]

    dico = {}
    idx_docs = []

    cpt = 0

    for doc in range(len(X)):
        idx = []
        for word in mat_contents["words"][0, doc][0]:
            if word[0] not in dico.keys():
                dico[word[0]] = cpt
                idx.append(cpt)
                cpt += 1
            else:
                idx.append(dico[word[0]])

        idx_docs.append(idx)
        
    return X, y, w, idx_train, idx_test, idx_docs, len(dico)


def get_movies():
    X = np.load("./data/X_movie.npy", allow_pickle=True)
    w = np.load("./data/w_movie.npy", allow_pickle=True)
    y = np.loadtxt("./data/y_movie.csv")
    words = np.load("./data/words_movie.npy", allow_pickle=True)

    dataset = "movie"
    path = "./results_movie/"


    idx_train, idx_test = [], []

    np.random.seed(2023)
    seeds = np.random.randint(0, 1000, 5)

    for k in range(5):
        y_train, y_test, id_train, id_test = train_test_split(y, list(range(len(X))), shuffle=True, 
                                                              random_state=seeds[k], stratify=y)
        idx_train.append(id_train)
        idx_test.append(id_test)


    dico = {}
    idx_docs = []

    cpt = 0

    for doc in range(len(X)):
        idx = []
        for word in words[doc]:
            if word not in dico.keys():
                dico[word] = cpt
                idx.append(cpt)
                cpt += 1
            else:
                idx.append(dico[word])            
        idx_docs.append(idx)
        
    return X, y, w, np.array(idx_train), np.array(idx_test), idx_docs, len(dico)


def get_goodreads(task="likability"):
    X = np.load("./data/X_goodread.npy", allow_pickle=True)
    w = np.load("./data/w_goodread.npy", allow_pickle=True)
    
    if task == "likability":
        y = np.loadtxt("./data/y_likability_goodread.csv")
    elif task == "genre":
        y = np.loadtxt("./data/y_genre_goodread.csv")
    
    words = np.load("./data/words_goodread.npy", allow_pickle=True)

    path = "./results_goodreads/"
    dataset = "goodreads"

    idx_train, idx_test = [], []

    np.random.seed(2023)
    seeds = np.random.randint(0, 1000, 5)

    for k in range(5):
        y_train, y_test, id_train, id_test = train_test_split(y, list(range(len(X))), 
                                                              shuffle=True, random_state=seeds[k],
                                                              stratify=y)
        idx_train.append(id_train)
        idx_test.append(id_test)


    dico = {}
    idx_docs = []

    cpt = 0


    for doc in range(len(X)):
        idx = []
        for word in words[doc]:
            if word not in dico.keys():
                dico[word] = cpt
                idx.append(cpt)
                cpt += 1
            else:
                idx.append(dico[word])            
        idx_docs.append(idx)
        
    return X, y, w, np.array(idx_train), np.array(idx_test), idx_docs, len(dico)

