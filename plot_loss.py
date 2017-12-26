import os
import sys
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print('Requires path to loss file')
    sys.exit('Exiting')

losses_file = sys.argv[1]
if not os.path.isfile(losses_file):
    print('File %s does not exit' % losses_file)
    sys.exit('Exiting')

with open(losses_file, 'rb') as f:
    losses = pickle.load(f)

plt.plot(losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.savefig(losses_file + '.png')
