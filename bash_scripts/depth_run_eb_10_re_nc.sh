#/bin/sh

cd ../

ipython depth_run.py microsoft 10 resample nc eb l rf tree 10000 1
ipython depth_run.py microsoft 10 resample nc eb l rf all 10000 1
ipython depth_run.py microsoft 10 resample nc eb l r tree 10000 1
ipython depth_run.py microsoft 10 resample nc eb l r all 10000 1
ipython depth_run.py microsoft 10 resample nc eb p rf tree 10000 1
ipython depth_run.py microsoft 10 resample nc eb p rf all 10000 1
ipython depth_run.py microsoft 10 resample nc eb p r tree 10000 1
ipython depth_run.py microsoft 10 resample nc eb p r all 10000 1
