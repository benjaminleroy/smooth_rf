#/bin/sh

cd ../

ipython depth_run.py 10 oracle c lb l
ipython depth_run.py 10 oracle c lb p
ipython depth_run.py 10 oracle nc lb l
ipython depth_run.py 10 oracle nc lb p
ipython depth_run.py 300 oracle c lb l
ipython depth_run.py 300 oracle c lb p
ipython depth_run.py 300 oracle nc lb l
ipython depth_run.py 300 oracle nc lb p
