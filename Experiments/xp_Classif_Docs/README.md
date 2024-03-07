## Instructions to get the data

- Download the BBCSport and Twitter dataset from https://github.com/mkusner/wmd/tree/master
- For the next datasets, first download GoogleNews-vectors-negative300.bin.gz from e.g. https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/view?resourcekey=0-wjGZdNAUop6WykTtMip30g
- For MovieReviews, first download the dataset from http://www.cs.cornell.edu/people/pabo/movie-review-data/ and then launch
```
python dataset.py --dataset movie
```
- For the goodreads dataset, first download the dataset from https://ritual.uh.edu/multi_task_book_success_2017/ and then launch
```
python dataset.py --dataset goodreads
```


## Experiments

The list of experiments can be found in the file `experiments.sh`.

- First, run `xp_pretrain_A.py` with the right parameters to get the matrix A learned with the NCA algorithm.
```
python xp_pretrain_A.py --dataset "BBC" --d 30
```
- Then, run `xp_doc_classif_wcd.py` with the right parameters to get the matrices of distance:
```
python xp_doc_classif_wcd.py --dataset "BBC" --loss "w2" --d 30 --ntry 1
```
<!-- - Run `xp_doc_classif.py` with the right parameters to get the matrices of distance:
```
python xp_doc_classif.py --pbar --loss "rsw" --n_projs 500 --dataset "BBC" --ntry 1 --rho1 0.0005 --rho2 0.0005 --inner_iter 10
``` -->
- Get the results of kNN using the functions in `utils_knn.py`.

