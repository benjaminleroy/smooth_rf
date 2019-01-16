#/bin/sh

cd ../

ipython depth_run.py microsoft 10 oob c eb p rf tree 10000 .1
ipython depth_run.py microsoft 10 oob c eb p rf tree 10000 1
ipython depth_run.py microsoft 10 oob c eb p rf tree 10000 10
ipython depth_run.py microsoft 10 oob c eb p rf tree 10000 100


ipython depth_run.py microsoft 300 oob c eb p rf tree 10000 .1
ipython depth_run.py microsoft 300 oob c eb p rf tree 10000 1
ipython depth_run.py microsoft 300 oob c eb p rf tree 10000 10
ipython depth_run.py microsoft 300 oob c eb p rf tree 10000 100
