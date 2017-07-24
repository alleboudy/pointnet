import os
import sys
import numpy as np
import h5py
from random import shuffle
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--input', default='', help='.h5 file to fix')
parser.add_argument('--output', default='', help='path to fixed file')
FLAGS = parser.parse_args()


labelsMap = dict({"bird":0,"can":1,"cracker":2,"house":3,"shoe":4})
inv_map = {v: k for k, v in labelsMap.items()}
oldMap = dict({"bird":0,"bond":1,"can":2,"cracker":3,"house":4,"shoe":5,"teapot":6})
old_inv_map = {v: k for k, v in oldMap.items()}


def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

# Write numpy array data and label to h5_filename
def save_h5_data_label_normal(h5_filename, data, label,  
        data_dtype='float32', label_dtype='uint8'):
    h5_fout = h5py.File(h5_filename)
    h5_fout.create_dataset(
            'data', data=data,
            compression='gzip', compression_opts=4,
            dtype=data_dtype)
    h5_fout.create_dataset(
            'label', data=label,
            compression='gzip', compression_opts=1,
            dtype=label_dtype)
    h5_fout.close()

file2fix=FLAGS.input
fixedfilelocation=FLAGS.output
data,labels = load_h5(file2fix)
print(data.shape)
print(labels.shape)
newdata=[]
newlabels=[]
for i in range(len(labels)):
	if labels[i] not in [1,6]:
		newdata.append(data[i])
		if labels[i]==0:
			newlabels.append([0])
		else:	
			newlabels.append(labels[i]-1)
	



save_h5_data_label_normal(fixedfilelocation,np.asarray(newdata),np.asarray(newlabels))