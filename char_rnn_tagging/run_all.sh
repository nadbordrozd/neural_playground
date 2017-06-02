#!/usr/bin/env bash
bash get_data.sh

python prepare_input_files.py data/austen 'austen.txt' data/austen_clean
awk '{ print $0 > "data/austen_clean/part"++i".txt" }' RS='\n\n\n' data/austen_clean/austen.txt
rm data/austen_clean/austen.txt
python prepare_input_files.py data/scikit-learn/ '*.py' data/sklearn_clean

python train_test_split.py data/austen_clean/ 0.2
python train_test_split.py data/sklearn_clean/ 0.2

python train.py models/model_1 data/sklearn_clean/ data/austen_clean
python apply_tagger.py models_model_1 data/sklearn_clean/ data/austen_clean