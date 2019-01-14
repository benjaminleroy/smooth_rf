#/bin/sh

cd ../

ipython depth_run.py 10 oob nc eb l rf tree 10000
ipython depth_run.py 10 oob nc eb l rf all 10000
ipython depth_run.py 10 oob nc eb l r tree 10000
ipython depth_run.py 10 oob nc eb l r all 10000
ipython depth_run.py 10 oob nc eb p rf tree 10000
ipython depth_run.py 10 oob nc eb p rf all 10000
ipython depth_run.py 10 oob nc eb p r tree 10000
ipython depth_run.py 10 oob nc eb p r all 10000
