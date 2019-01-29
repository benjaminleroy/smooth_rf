#/bin/sh

cd ../

ipython depth_run.py microsoft 300 oracle c eb l rf tree 10000 1
ipython depth_run.py microsoft 300 oracle c eb l rf all 10000 1
ipython depth_run.py microsoft 300 oracle c eb l r tree 10000 1
ipython depth_run.py microsoft 300 oracle c eb l r all 10000 1
ipython depth_run.py microsoft 300 oracle c eb p rf tree 10000 1
ipython depth_run.py microsoft 300 oracle c eb p rf all 10000 1
ipython depth_run.py microsoft 300 oracle c eb p r tree 10000 1
ipython depth_run.py microsoft 300 oracle c eb p r all 10000 1
