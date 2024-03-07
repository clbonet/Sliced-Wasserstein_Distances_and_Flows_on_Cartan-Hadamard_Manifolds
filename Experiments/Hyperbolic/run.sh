#!/bin/bash

## For WNDs:
python xp_swfs.py --type_target "wnd" --target "center" --ntry 5 --lr 0.1 --n_epochs 201
python xp_swfs.py --type_target "wnd" --target "border" --ntry 5 --lr 0.5 --n_epochs 301

## For mixture of WNDs
python xp_swfs.py --type_target "mwnd" --target "center" --ntry 5 --lr 0.1 --n_epochs 201
python xp_swfs.py --type_target "mwnd" --target "border" --ntry 5 --lr 0.5 --n_epochs 301

