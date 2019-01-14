#/bin/sh

cd ../

ipython depth_run.py 300 resample c eb l rf tree 10000
ipython depth_run.py 300 resample c eb l rf all 10000
ipython depth_run.py 300 resample c eb l r tree 10000
ipython depth_run.py 300 resample c eb l r all 10000
ipython depth_run.py 300 resample c eb p rf tree 10000
ipython depth_run.py 300 resample c eb p rf all 10000
ipython depth_run.py 300 resample c eb p r tree 10000
ipython depth_run.py 300 resample c eb p r all 10000
