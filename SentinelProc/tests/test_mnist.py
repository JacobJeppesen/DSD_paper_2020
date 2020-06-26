import pytest
import sys 

sys.path.append('..')
from app.src.mnist import train_mnist

deterministic_training = True

if deterministic_training:
  # Code snippet for reproducibility in Keras (https://stackoverflow.com/questions/48631576/reproducible-results-using-keras-with-tensorflow-backend)
  # Note: Perfect reproducibility is not guaranteed when using GPUs for training (see link above)
  # =================================================================================================================================================
  # Seed value (can actually be different for each attribution step)
  seed_value= 0

  # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
  import os
  os.environ['PYTHONHASHSEED']=str(seed_value)

  # 2. Set `python` built-in pseudo-random generator at a fixed value
  import random
  random.seed(seed_value)

  # 3. Set `numpy` pseudo-random generator at a fixed value
  import numpy as np
  np.random.seed(seed_value)

  # 4. Set `tensorflow` pseudo-random generator at a fixed value
  import tensorflow as tf
  tf.set_random_seed(seed_value)

  # 5. Configure a new global `tensorflow` session
  from keras import backend as K
  session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
  sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
  K.set_session(sess)
  # =================================================================================================================================================


def test_training_mnist():
    test_accuracy = train_mnist()
    assert test_accuracy == pytest.approx(0.9738, 0.01)
