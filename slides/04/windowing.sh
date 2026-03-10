#!/bin/sh

python3 windowing.py
for f in windowing*.svg; do
  svgreduce $f $f.red
  mv $f.red $f
  > $f.ref
done
