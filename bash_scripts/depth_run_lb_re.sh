#/bin/sh

cd ../

ipython depth_run.py microsoft 10 resample c lb l
ipython depth_run.py microsoft 10 resample c lb p
ipython depth_run.py microsoft 10 resample nc lb l
ipython depth_run.py microsoft 10 resample nc lb p
ipython depth_run.py microsoft 300 resample c lb l
ipython depth_run.py microsoft 300 resample c lb p
ipython depth_run.py microsoft 300 resample nc lb l
ipython depth_run.py microsoft 300 resample nc lb p
