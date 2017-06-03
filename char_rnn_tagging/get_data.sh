#!/usr/bin/env bash
mkdir -p data/austen
cd data
wget http://www.gutenberg.org/ebooks/31100.txt.utf-8
mv 31100.txt.utf-8 austen/austen.txt

git clone https://github.com/scikit-learn/scikit-learn.git
git clone https://github.com/scalaz/scalaz.git