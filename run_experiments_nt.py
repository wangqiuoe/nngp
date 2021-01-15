# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import csv
import os.path
import time
from sklearn.utils import shuffle

import pandas as pd
import numpy as np
import tensorflow as tf

import gpr
import load_dataset
import nngp


class InfoFilter(logging.Filter):
    def filter(self, record):
        return record.levelno == logging.INFO


log = logging.getLogger('tensorflow')
log.setLevel(logging.INFO)
log.addFilter(InfoFilter())
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# create file handler which logs even debug messages
fh = logging.FileHandler('tensorflow.log')
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
log.addHandler(fh)

tf.logging.set_verbosity(tf.logging.INFO)


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('hparams', '',
                    'Comma separated list of name=value hyperparameter pairs to'
                    'override the default setting.')

#add by qi
flags.DEFINE_string('data_path','./Folds5x2_pp.xlsx', 'datapath')
flags.DEFINE_float('train_test_split', 0.5, 'train_test_split_ratio')
#add by qi

flags.DEFINE_string('experiment_dir', './tmp/nngp',
                    'Directory to put the experiment results.')
flags.DEFINE_integer('seed', 1234, 'Random number seed for data shuffling')
flags.DEFINE_string('dataset', 'qi',
                    'Which dataset to use ["mnist","qi"]')
flags.DEFINE_float('noise_var', 1e-4,
                    'noise_var add to variance of train kernal matrix')
flags.DEFINE_string('nonlinearity', 'relu',
        'activation function: ["relu", "tanh"]')
flags.DEFINE_integer('depth', 2, 'number of depths of NN')


def normalize_y(train):
    mu = np.mean(train,0)
    std = np.std(train,0)
    train = (train-mu)/std
    return train,mu,std

def load_data():
    df = pd.read_excel(FLAGS.data_path)
    x = df[['AT','V','AP','RH']].to_numpy()
    y = df[['PE']].to_numpy()
    x, y = shuffle(x, y, random_state=1)
    len_train = int(x.shape[0]*FLAGS.train_test_split)
    train_x = x[:len_train,:]
    train_y = y[:len_train,:]
    test_x = x[len_train:,:]
    test_y = y[len_train:,:]
    train_y_normal,mu_y,std_y = normalize_y(train_y)
    return train_x, train_y, test_x, test_y,mu_y,std_y, train_y_normal

def do_eval_qi(model, x_data, y_data, y_mu,y_std, save_pred=False):
  """Run evaluation."""

  gp_prediction = model.predict(x_data)
  y_pred = gp_prediction* y_std + y_mu
  rmse = np.sqrt(np.mean((y_data - y_pred)**2))
  tf.logging.info('RMSE: %.8f'%rmse)

  if save_pred:
    with tf.gfile.Open(
        os.path.join(FLAGS.experiment_dir, 'gp_prediction_stats.npy'),
        'w') as f:
      np.save(f, y_pred)

  return rmse

def run_nngp_eval(run_dir):
  """Runs experiments."""
  tf.logging.info('Hyperparameters')
  tf.logging.info('---------------------')
  tf.logging.info(hparams)
  tf.logging.info('---------------------')
  tf.logging.info('Loading data')

  # Get the sets of images and labels for training, validation, and
  # # test on dataset.
  if FLAGS.dataset == 'qi':
    train_image, train_label, test_image,test_label,y_mu,y_std,train_label_normal = load_data()
  else:
    raise NotImplementedError
  tf.logging.info('train X shape: %s' %(str(train_image.shape)))
  tf.logging.info('train y shape: %s' %(str(train_label.shape)))
  tf.logging.info('test X shape: %s' %(str(test_image.shape)))
  tf.logging.info('test y shape: %s' %(str(test_label.shape)))

  #return train_image, train_label, valid_image, valid_label, test_image,test_label, mu_y,std_y

  tf.logging.info('Building Model')

  model = nt.NNGPRegression(train_image, train_label_normal, noise_var = FLAGS.noise_var,nonlinearity=FLAGS.nonlinearity ,n_depth = FLAGS.depth)

  start_time = time.time()
  tf.logging.info('Evaluating Train set')
  rmse_train = do_eval_qi(model, train_image,train_label, y_mu,y_std)
  tf.logging.info('Evaluation of training set (%d examples) took '
                    '%.3f secs'%(
                        (train_image.shape[0]),
                        time.time() - start_time))

  start_time = time.time()
  tf.logging.info('Evaluating test set')
  rmse_test = do_eval_qi(
      model,
      test_image,
      test_label,
      y_mu,y_std,
      save_pred=False)
  tf.logging.info('Evaluation of test set (%d examples) took %.3f secs'%(
      test_image.shape[0], time.time() - start_time))

  metrics = {
      'train_rmse': float(rmse_train),
      'test_rmse': float(rmse_test),
  }

  record_results = [
      	FLAGS.train_test_split, FLAGS.nonlinearity, FLAGS.depth, FLAGS.noise_var, rmse_train, rmse_test,
        ]
  # Store data
  result_file = os.path.join(run_dir, 'results.csv')
  with tf.gfile.Open(result_file, 'a') as f:
    filewriter = csv.writer(f)
    filewriter.writerow(record_results)

  return metrics


def main(argv):
  run_nngp_eval(FLAGS.experiment_dir)


if __name__ == '__main__':
  tf.app.run(main)

