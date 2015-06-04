#!/bin/bash

export id=$1
pct_train=$2
pct_val=$3
error=$4
seconds=$5

mkdir -p data
mkdir -p embeddings
mkdir -p logs
mkdir -p fitted
mkdir -p snapshots

# dataset: 1-4 letters per clause, 2-6 clauses
python generate_grammar-1.py data/x_all.txt.${id} data/y_all.txt.${id} 0 1 4 2 6

# partition dataset into train, validate, test
python partition.py data/x_all.txt.${id} data/y_all.txt.${id} $pct_train data/x_train.txt.${id} data/y_train.txt.${id} $pct_val data/x_val.txt.${id} data/y_val.txt.${id} 'r' data/x_test.txt.${id} data/y_test.txt.${id}

# create rc
envsubst < grammar.rc > grammar.rc.${id}

# run program
python ../../translate.py "@grammar.rc.${id}" --error $error --seconds $seconds
