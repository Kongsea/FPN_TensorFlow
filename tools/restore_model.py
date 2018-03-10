# -*- coding:utf-8 -*-

from __future__ import absolute_import, division, print_function

import os

import tensorflow as tf
import tensorflow.contrib.slim as slim

from libs.configs import cfgs
from libs.networks.network_factory import get_flags_byname

RESTORE_FROM_RPN = True
FLAGS = get_flags_byname(cfgs.NET_NAME)


def get_restorer(test=True, checkpoint_path=None):

  if test:
    assert checkpoint_path != None, "When testing, checkpoint must be set."
  else:
    checkpoint_path = tf.train.latest_checkpoint(
        os.path.join(FLAGS.trained_checkpoint, cfgs.VERSION))

  if checkpoint_path != None:
    if RESTORE_FROM_RPN:
      print('___restore from rpn___')
      model_variables = slim.get_model_variables()
      restore_variables = [var for var in model_variables if not var.name.startswith(
          'Fast_Rcnn')] + [tf.train.get_or_create_global_step()]
      restorer = tf.train.Saver(restore_variables)
    else:
      restorer = tf.train.Saver()
    print("model restore from :", checkpoint_path)
  else:
    checkpoint_path = FLAGS.pretrained_model_path
    print("model restore from pretrained mode, path is:", checkpoint_path)

    model_variables = slim.get_model_variables()

    restore_variables = [var for var in model_variables if
                         (var.name.startswith(cfgs.NET_NAME)
                          and not var.name.startswith('{}/logits'.format(cfgs.NET_NAME)))]
    restorer = tf.train.Saver(restore_variables)

  return restorer, checkpoint_path
