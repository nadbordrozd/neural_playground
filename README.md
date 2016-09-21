This repository contains a set of command line tools for playing with neural networks. So far this means text generation with char-RNN. At the core this is nothing more than the [keras lstm_text_generation.py](https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py) example, just more fleshed out and with some utilities around it. I will add some image generation scripts when I have the time. 

A typical workflow consists of 
1. getting the data 
2. training the model
3. using the model to generate text

### Getting the data
Start with downloading some fun textual datasets to work with. Scripts in `getting_data` do exactly that. 
```bash
getting_Data/hadoop.sh data/lots_of_java.java
```
will clone the hadoop codebase from github, concatenate all the java files and put them in `data/lots_of_java.java`.
```bash
getting_Data/sklearn.sh data/all_the_python.py
```
will do the same for scikit-learn. Other scripts follow the same interface and download the dataset of Enron emails (this is a big one!), collected novels of Jane Austen, scalaz codebase and so on. 

Also included is a youtube comments scraper: 
```bash
getting_data/youtube_comments.py your_yt_api_key "flat earth" data/flat_earth_comments.txt \
    --max_videos=500 \
    --threads=50
```
will scrape all comments from top 500 youtube videos matching the query "flat earth" and put them in `data/flat_earth_comments.txt`. It's limited to top level comments.
 
Having downloaded the dataset let's split it into train and test sets:
```bash
getting_data/train_test_split.sh data/flat_earth_comments.txt 0.1 data/flat_earth
```
will put the firs 90% of the lines of `flat_earth_comments.txt` file in `data/flat_earth/train` and the remaining 10% in `data/flat_earth/test`. 

### Training the model
Simply run:
```bash
./train_text_gen.py \
    --train_test_path data/flat_earth \
    --model_dir models/flat_earth_1 \
    --maxlen 160 \
    --lstm_size 128 \
    --dropout 0.2 \
    --batch_size 1024 \
    --max_epochs 2 \
    --layers 3
```

This will train a network with 3 LSTM layers (128 units each) followed by a dense layer and with 20% dropout after each layer. The network is trained on `data/flat_earth/train` with `data/flat_earth/test` as test set. The model is saved after every epoch in `models/flat_earth_1/epoch_%%%%%`. The network is trained to predict the next character in based of the previous `maxlen` characters and the training examples are generated from the input text by sliding a window of `maxlen` chars by `step` characters to go from one example to the next. `step` defaults to 1, and I haven't researched how big of a difference it makes if you set it to a higher number (but it will definitely make the epoch go faster!). 
 
 `max_epochs` is 2 in this example but the default is 10000 because in my typical workflow I always stop it manually. If the dataset and the network are big enough you could easily train it for weeks and still see improvements in validation loss (which is printed to stdout). So it's best to just let it run indefinitely and only kill it when you run out of patience or notice that loss is not improving significantly anymore. 
 
You can always restart training of a model by running the same command: 
```bash
./train_text_gen.py \
    --train_test_path data/flat_earth \
    --model_dir models/flat_earth_1 \
    --batch_size 1024
```

it will pick up from the latest saved version. `--lstm_size`, `--dropout`, `--layers` parameters are in this case not necessary since these things are saved with the model. 

NOTE:
If you go overboard with the size of your network - which is easy to do - you will get a very long error message where one of the lines will look like this: 
`MemoryError: Error allocating 2717908992 bytes of device memory (out of memory).`
This means a single batch doesn't fit in memory and you have to decrease batch size (every epoch will take longer). If even batch size of 1 doesn't solve the problem then your network itself must be to big to fit in your CPU's or GPU's memory and you have to cut back on `lstm_size` or `layers`. 
### Generating text

Finally, the fun part. To generate 10000 characters with the model we just trained, run:

```bash
./generate_text.py \
    --model_path models/flat_earth_1 \
    --load_latest \
    --seed "`head data/flat_earth/train -n 20`" \
    --chars 10000 \
    --diversity 1 \
    --out_file data/generated_flat_earth.txt
```

Here the first 20 lines of the train file will e used as seed for the generator. Generated text will be put in `data/generated_flat_earth.txt` in addition to being printed to standard output one character at a time as it is being generated (fun to watch!).
