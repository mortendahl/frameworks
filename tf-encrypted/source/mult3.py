from __future__ import absolute_import

import tensorflow as tf
import tf_encrypted as tfe

from data_loading import load_from_textfile

# tfe.set_config(tfe.RemoteConfig({
#   'server0': 'localhost:0000'
# }))

# load inputs
num_rows = 9
x0 = tfe.define_private_input('server0', lambda: load_from_textfile('source/InputData/mult3-p0', (num_rows,)))
x1 = tfe.define_private_input('server1', lambda: load_from_textfile('source/InputData/mult3-p1', (num_rows,)))
x2 = tfe.define_private_input('server1', lambda: load_from_textfile('source/InputData/mult3-p2', (num_rows,)))

# compute product
y = x0 * x1 * x2

with tfe.Session() as sess:
  # reveal output and print
  print(sess.run(y.reveal()))
