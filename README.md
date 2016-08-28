### fun datasets to train char-RNNs on
Scripts in `getting_data` do exactly that.
`bash getting_data/hadoop.sh data/lots_of_java.java`
Clones the hadoop repository from github, concatenates all java files in it and puts then in `data/lots_of_java.java`  
`bash getting_data/sklearn.sh data/lots_of_python.py`  
will do the same for scikit-learn and python  
`bash getting_data/enron.sh data/enron.txt`  
downloads the enron email dataset


```bash
bash python getting_data/youtube_comments.py your_yt_api_key "flat earth" data/flat_earth_comments.txt --max_videos=500
```
This will scrape all coments from top 500 youtube videos matching the query "flat earth" and put them in `data/flat_earth_comments.txt`