import tensorflow as tf

# helper method for loading a tensor from a text file;
# this will get executed locally on each player
def load_from_textfile(filename, input_size):
  raw_content = tf.io.read_file(filename)
  raw_content = tf.reshape(raw_content, shape=(1,))

  values_as_strings = tf.strings.split(raw_content, '\n').values
  values_as_strings = tf.reshape(values_as_strings, shape=(1, input_size))

  values_as_numbers = tf.strings.to_number(values_as_strings)

  return tf.Print(
    values_as_numbers,
    [values_as_numbers],
    summarize=input_size,
    message="Loaded from '{}': ".format(filename)
  )