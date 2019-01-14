#/bin/sh

cd ../

ipython depth_run.py 10 oob c lb l
ipython depth_run.py 10 oob c lb p
ipython depth_run.py 10 oob nc lb l
ipython depth_run.py 10 oob nc lb p
ipython depth_run.py 300 oob c lb l
ipython depth_run.py 300 oob c lb p
ipython depth_run.py 300 oob nc lb l
ipython depth_run.py 300 oob nc lb p
