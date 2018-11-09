import numpy as np
import tensorflow as tf

# helper method for loading a tensor from a text file;
# this will get executed locally on each player
def load_from_textfile(filename, shape):
  raw_content = tf.io.read_file(filename)
  raw_content = tf.reshape(raw_content, shape=(1,))

  values_as_strings = tf.strings.split(raw_content, '\n').values
  values_as_strings = tf.reshape(values_as_strings, shape=shape)

  values_as_numbers = tf.strings.to_number(values_as_strings)

  return tf.Print(
    values_as_numbers,
    [values_as_numbers],
    summarize=np.prod(shape),
    message="Loaded from '{}': ".format(filename)
  )