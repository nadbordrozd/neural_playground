Here be a series of command line tools for experimenting with char-RNN for tagging text. 

- `get_data.sh` downloads a few textual datasets to play with - scikit-learn github repo, 
collected works of Jane Austen, etc.
- `prepare_input_files.py` cleans up files from a given dataset and puts them in one place.
- `train_test_split.py` takes files prepared in the previous step and partitions them into train 
and test directories at random.
- `train.py` takes two sets of files e.g. sklearn code and Jane Austen novels, splices them 
together and trains a char-RNN to recognize for every character, which source did it come from. 
(see below). 
- `apply_tagger.py` takes model trained in the previous step and applies it to the test set. 
Saves texts, predictions and true labels.
- `plot_predictions.py` takes output of the previous step and produces html files visualising the
 predictions. Model prediction is represented with color and true label is 
represented by different fonts.
- `run_all.sh` runs all the above steps.




