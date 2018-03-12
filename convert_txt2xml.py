#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import os
import xml.etree.cElementTree as ET
import cv2
import PIL
import copy
from tqdm import tqdm

from libs.configs import cfgs

template_file = 'sample.xml'
target_dir = os.path.join(cfgs.ROOT_PATH, 'data/{}/Annotations/'.format(cfgs.DATASET_NAME))
image_dir = os.path.join(cfgs.ROOT_PATH, 'data/{}/JPEGImages/'.format(cfgs.DATASET_NAME))
anno_dir = os.path.join(cfgs.ROOT_PATH, 'data/{}/annotations/'.format(cfgs.DATASET_NAME))

anno_files = [os.path.join(anno_dir, f) for f in os.listdir(
    anno_dir) if f.endswith('.txt') and not f.startswith('.')]

if not os.path.exists(target_dir):
  os.makedirs(target_dir)

for af in tqdm(anno_files):
  if not os.path.exists(af):
    continue

  with open(af) as f:
    anno_lines = [f.strip() for f in f.readlines()]

  if not anno_lines:
    continue

  image_file = af.rpartition('/')[-1].replace('txt', 'jpg')

  tree = ET.parse(template_file)
  root = tree.getroot()

  # filename
  root.find('filename').text = image_file
  # size
  sz = root.find('size')
  im = cv2.imread(image_dir + image_file)
  sz.find('height').text = str(im.shape[0])
  sz.find('width').text = str(im.shape[1])
  sz.find('depth').text = str(im.shape[2])

  # object
  obj_ori = root.find('object')
  root.remove(obj_ori)

  for al in anno_lines:
    if al.startswith('rotate'):
      continue
    bb_info = al.split()

    x_1 = int(bb_info[1])
    y_1 = int(bb_info[2])
    x_2 = int(bb_info[3])
    y_2 = int(bb_info[4])

    obj = copy.deepcopy(obj_ori)

    obj.find('name').text = bb_info[0].decode('utf-8')
    bb = obj.find('bndbox')
    bb.find('xmin').text = str(x_1)
    bb.find('ymin').text = str(y_1)
    bb.find('xmax').text = str(x_2)
    bb.find('ymax').text = str(y_2)

    root.append(obj)

  xml_file = image_file.replace('jpg', 'xml')

  tree.write(target_dir + xml_file, encoding='utf-8', xml_declaration=True)

print('Done')
