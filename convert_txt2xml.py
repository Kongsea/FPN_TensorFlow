#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import xml.etree.cElementTree as ET
import cv2
import PIL
import copy

template_file = 'anno.xml'
# 生成的xml文件保存的路径
target_dir = '/home/konghaiyang/lm/FPN_Tensorflow/data/icecream/Annotations/'
# 所有图片的路径
image_dir = '/home/konghaiyang/lm/FPN_Tensorflow/data/icecream/JPEGImages/'

anno_dir = '/home/konghaiyang/data/icecream/annotations/'

anno_files = [os.path.join(anno_dir, f) for f in os.listdir(anno_dir) if f.endswith('.txt')]

for af in anno_files:
  with open(af) as f:
    anno_lines = [f.strip() for f in f.readlines()]

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

  if im.shape[1] == 720:
    print(af)

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

  print xml_file

print 'Done'
