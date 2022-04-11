#!/bin/bash

set -e

if [ ! -d venv ]; then
  apt-get install virtualenv
  virtualenv -p venv
fi

source venv/bin/activate
pip install -r requirements.txt

export PYTHONPATH=`pwd`
cd linearcode

sudo sh -c 'echo -1 >/proc/sys/kernel/perf_event_paranoid'
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
python correction_overhead.py