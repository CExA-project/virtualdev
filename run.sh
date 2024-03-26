#!/bin/sh
cd "$(dirname "$0")"
mpic++ -rdynamic -g -std=c++23 -o vdev vdev.cxx && exec mpirun -n 2 ./vdev
