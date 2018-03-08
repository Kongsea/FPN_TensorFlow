#!/usr/bin/python
# -*- coding: utf-8 -*-
import os

anno_dir = 'data/layer/annotations'

anno_files = [os.path.join(anno_dir, f) for f in os.listdir(anno_dir) if f.endswith('txt')]

classes = []

for af in anno_files:
  with open(af) as f:
    cls = [line.strip().split()[0] for line in f.readlines()]

  classes.extend(cls)

classes = list(set(classes))

with open('classes.txt', 'w') as f:
  for i, cls in enumerate(classes, 1):
    if 'rotate' in cls:
      continue
    f.write('{}\n'.format(cls))
