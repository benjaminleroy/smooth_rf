#/bin/sh

cd ../main/

ipython depth_run.py microsoft 10 oob c eb p standard rf tree 10000 1
ipython depth_run.py microsoft 10 oob c eb p max rf tree 10000 1
ipython depth_run.py microsoft 10 oob c eb p min rf tree 10000 1
