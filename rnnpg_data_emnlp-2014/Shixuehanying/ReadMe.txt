Shixuehanying.txt and Shixuehanying.txt.jian are crawled from http://cls.hs.yzu.edu.tw/tang/MakePoem/SSHY.htm
This program and the crawled data can only be used for research purpose.


run the following commands to get the shixuehanying.txt (the traditional Chinese version) and shixuehanying.txt.jian (the simplified Chinese version) 
./before_u_go.sh
./ShixuehanyingCrawler.py 1
./ShixuehanyingCrawler.py 2
./ShixuehanyingCrawler.py 3

Please make sure these three commands end sucessfully. The second command sometimes fails due to the network connection. You need to restart it from where it failed. ./ShixuehanyingCrawler.py 2 -b LAST_PAGE_ID   See ./ShixuehanyingCrawler.py -h for more details. 
