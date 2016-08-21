# how to get lots of java files
git clone https://github.com/apache/hadoop.git
cat $(find hadoop -name "*.java" -type f) > all_the_java.java

# how to get lots of youtube comments
python youtube_comments.py yt_api_key_goes_here "flat earth" data/flat_earth_comments.txt