#/bin/sh

cd ../

ipython depth_run.py online_news 10 oob c eb p rf tree 10000 1
ipython depth_run.py online_news 300 oob c eb p rf tree 10000 1

