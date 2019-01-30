#/bin/sh

cd ../main/

ipython depth_run.py microsoft 10 oob c eb l standard rf tree 10000 1
ipython depth_run.py microsoft 10 oob c eb l max rf tree 10000 1
ipython depth_run.py microsoft 10 oob c eb l min rf tree 10000 1
