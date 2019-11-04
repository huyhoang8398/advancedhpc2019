#!/bin/bash
if ls build* 1> /dev/null 2>&1; then
  rm -rf build &&
  echo "Building..."
  mkdir build &&
  cd build &&
  cmake .. &&
  make -j
else
  echo "Building..." &&
  mkdir build &&
  cd build &&
  cmake .. &&
  make -j 
fi