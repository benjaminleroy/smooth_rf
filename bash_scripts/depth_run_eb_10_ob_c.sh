#/bin/sh

cd ../

ipython depth_run.py microsoft 10 oob c eb l rf tree 10000 1
ipython depth_run.py microsoft 10 oob c eb l rf all 10000 1
ipython depth_run.py microsoft 10 oob c eb l r tree 10000 1
ipython depth_run.py microsoft 10 oob c eb l r all 10000 1
ipython depth_run.py microsoft 10 oob c eb p rf tree 10000 1
ipython depth_run.py microsoft 10 oob c eb p rf all 10000 1
ipython depth_run.py microsoft 10 oob c eb p r tree 10000 1
ipython depth_run.py microsoft 10 oob c eb p r all 10000 1
