## Perfs xp
python xp_pretrain_A.py --dataset "BBC" --d 30
python xp_doc_classif_wcd.py --dataset "BBC" --loss "w2" --d 30 --ntry 1
python xp_doc_classif_wcd.py --dataset "BBC" --loss "sw" --d 30 --ntry 3 --n_projs 500

python xp_doc_classif_wmd.py --dataset "BBC" --loss "sw" --d 30 --ntry 3
python xp_doc_classif_wmd.py --dataset "BBC" --loss "w2" --d 30 --ntry 1


python xp_pretrain_A.py --dataset "movie" --d 30 --lr 1e-4 --n_epochs 100000
python xp_doc_classif_wcd.py --dataset "movie" --loss "w2" --d 30 --ntry 1
python xp_doc_classif_wcd.py --dataset "movie" --loss "sw" --d 30 --ntry 3 --n_projs 500

python xp_doc_classif_wmd.py --dataset "movie" --loss "sw" --d 30 --ntry 3
python xp_doc_classif_wmd.py --dataset "movie" --loss "w2" --d 30 --ntry 1


python xp_pretrain_A.py --dataset "goodreads_genre" --d 30 --lr 1e-3 --n_epochs 100000
python xp_doc_classif_wcd.py --dataset "goodreads_genre" --loss "sw" --d 30 --ntry 3 --n_projs 500
python xp_doc_classif_wcd.py --dataset "goodreads_genre" --loss "w2" --d 30 --ntry 1

python xp_doc_classif_wmd.py --dataset "goodreads_genre" --loss "sw" --d 30 --ntry 3
python xp_doc_classif_wmd.py --dataset "goodreads_genre" --loss "w2" --d 30 --ntry 1


## Runtime xp

python xp_doc_classif_wcd_runtime.py --dataset "BBC" --loss "w2" --d 30 --ntry 1
python xp_doc_classif_wcd_runtime.py --dataset "BBC" --loss "sw" --d 30 --ntry 1 --n_projs 500

python xp_doc_classif_wcd_runtime.py --dataset "movie" --loss "w2" --d 30 --ntry 1
python xp_doc_classif_wcd_runtime.py --dataset "movie" --loss "sw" --d 30 --ntry 1 --n_projs 500

python xp_doc_classif_wcd_runtime.py --dataset "goodreads_like" --loss "sw" --d 30 --ntry 1 --n_projs 500
python xp_doc_classif_wcd_runtime.py --dataset "goodreads_like" --loss "w2" --d 30 --ntry 1