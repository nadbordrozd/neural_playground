# how to get lots of java files
git clone https://github.com/apache/hadoop.git
cat $(find hadoop -name "*.java" -type f) > all_the_java.java