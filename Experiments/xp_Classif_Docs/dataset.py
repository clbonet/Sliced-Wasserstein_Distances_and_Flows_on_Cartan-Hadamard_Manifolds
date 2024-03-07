import glob
import gensim
import yaml
import argparse

import numpy as np

from gensim.parsing.preprocessing import remove_stopwords, strip_punctuation, stem_text
from yaml.loader import SafeLoader


def get_embedding(doc_train, doc_test):
    model = gensim.models.KeyedVectors.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin', binary=True)
    
    X_train = []
    w_train = []

    X_test = []
    w_test = []
    
    L_words_train = []
    L_words_test = []

    for file in doc_train:
        f = open(file, "r")
        raw_txt = f.read()
        txt = stem_text(strip_punctuation(remove_stopwords(raw_txt))).split()

        L = []
        L_w = []
        L_words = []
        for word in set(txt):
            try:
                c = txt.count(word)
                L.append(model[word])
                L_w.append(c)
                L_words.append(word)
            except:
                pass

        X_train.append(np.array(L).T)
        w_train.append([L_w])
        L_words_train.append(np.array(L_words))


    for file in doc_test:
        f = open(file, "r")
        raw_txt = f.read()
        txt = stem_text(strip_punctuation(remove_stopwords(raw_txt))).split()

        L = []
        L_w = []
        L_words = []
        for word in set(txt):
            try:
                c = txt.count(word)
                L.append(model[word])
                L_w.append(c)
                L_words.append(word)
            except:
                pass

        X_test.append(np.array(L).T)
        w_test.append([L_w])
        L_words_test.append(np.array(L_words))
        
    return X_train, w_train, X_test, w_test, L_words_train, L_words_test


def get_movie_review():
    """
        Get the dataset from http://www.cs.cornell.edu/people/pabo/movie-review-data/
    """
    doc_train = []
    doc_train_name = []
    doc_test = []
    doc_test_name = []

    for filename in glob.iglob("./data/review_polarity/train/*", recursive=True):
        doc_train.append(filename)
        doc_train_name.append(filename.split("/")[-1])

    for filename in glob.iglob("./data/review_polarity/test/*", recursive=True):
        doc_test.append(filename)
        doc_test_name.append(filename.split("/")[-1])

    labels_train = np.loadtxt("./data/review_polarity/labels_train", delimiter=",", dtype=str)
    labels_test = np.loadtxt("./data/review_polarity/labels_test", delimiter=",", dtype=str)

    y_train = []
    y_test = []

    for file in doc_train_name:
        idx = list(labels_train[:,0]).index(file)
        y_train.append(int(labels_train[idx, 1]))

    for file in doc_test_name:
        idx = list(labels_test[:,0]).index(file)
        y_test.append(int(labels_test[idx, 1]))
            
    X_train, w_train, X_test, w_test, L_words_train, L_words_test = get_embedding(doc_train, doc_test)
        
    X = X_train + X_test
    w = w_train + w_test
    y = y_train + y_test
    L_words = L_words_train + L_words_test
    
    return X, w, y, L_words


def get_goodreads():
    """
        Get the dataset from https://ritual.uh.edu/multi_task_book_success_2017/
        
        [1] Maharjan, Suraj, et al. "A multi-task approach to predict likability of books." Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 1, Long Papers. 2017.
    """    
    path = "./data/goodreads/train_test_split_goodreads.yaml"
    path_rating = "./data/goodreads/train_test_split_goodreads_avg_rating.yaml"

    f = open(path)
    data = yaml.load(f, Loader=SafeLoader)

    f = open(path_rating)
    data_rating = yaml.load(f, Loader=SafeLoader)


    y_train = []
    y_test = []

    y_genre_train = []
    y_genre_test = []

    dico_genre = {"Detective_and_mystery_stories": 0, "Drama": 1, "Fiction": 2, "Historical_fiction": 3,
                 "Love_stories": 4, "Poetry": 5, "Science_fiction": 6, "Short_stories": 7}

    doc_train = []
    doc_train_name = []
    doc_test = []
    doc_test_name = []

    for filename in glob.iglob("./data/goodreads/*/*/*", recursive=True):
        name = filename.split("/")[-1]
        label = filename.split("/")[-2]
        genre = filename.split("/")[-3]
        if name in data["test"]:
            doc_test_name.append(name)
            doc_test.append(filename)
            if label == "success":
                y_test.append(1)
            else:
                y_test.append(0)

            y_genre_test.append(dico_genre[genre])

        elif name in data["train"]:
            doc_train_name.append(name)
            doc_train.append(filename)
            if label == "success":
                y_train.append(1)
            else:
                y_train.append(0)

            y_genre_train.append(dico_genre[genre])
            
            
    X_train, w_train, X_test, w_test, L_words_train, L_words_test = get_embedding(doc_train, doc_test)
    
    X = X_train + X_test
    w = w_train + w_test
    y_likability = y_train + y_test
    y_genre = y_genre_train + y_genre_test
    L_words = L_words_train + L_words_test
    
    return X, w, y_likability, y_genre, L_words


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="movie", help="Which dataset to load")
    args = parser.parse_args()
    
    if args.dataset == "movie":
        X, w, y, L_words = get_movie_review()
        
        arr = np.empty(len(X), object)
        arr[:] = X
        np.save("./data/X_movie.npy", arr, allow_pickle=True)
        
        arr = np.empty(len(w), object)
        arr[:] = w
        np.save("./data/w_movie.npy", arr, allow_pickle=True)
        
        np.savetxt("./data/y_movie.csv", y)
        
        arr = np.empty(len(L_words), object)
        arr[:] = L_words
        np.save("./data/words_movie.npy", arr, allow_pickle=True)
        
    elif args.dataset == "goodreads":
        X, w, y_likability, y_genre, L_words = get_goodreads()
        
        arr = np.empty(len(X), object)
        arr[:] = X
        np.save("./data/X_goodread.npy", arr, allow_pickle=True)
        
        arr = np.empty(len(w), object)
        arr[:] = w
        np.save("./data/w_goodread.npy", arr, allow_pickle=True)
        
        np.savetxt("./data/y_likability_goodread.csv", y_likability)   
        np.savetxt("./data/y_genre_goodread.csv", y_genre)   
        
        arr = np.empty(len(L_words), object)
        arr[:] = L_words
        np.save("./data/words_goodread.npy", arr, allow_pickle=True)

