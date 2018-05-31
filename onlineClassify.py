import tensorflow as tf
from random import shuffle
import numpy as np
import time
import os
import sys
from os import listdir
from os.path import isfile, join
import argparse
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'log'))
import scipy.misc
import provider
import pc_util
import importlib
from plyfile import (PlyData, PlyElement, make2d, PlyParseError, PlyProperty)

from flask import Flask, jsonify, render_template, request
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


#parser = argparse.ArgumentParser()
#parser.add_argument('--ply_path', default='', help='ply file to classify')
#parser.add_argument('--batch_ply_path', default='', help='folder where .ply files exist, if set, will classify the files in one go')
#FLAGS = parser.parse_args()



BATCH_SIZE = 2
NUM_POINT = 2048
MODEL_PATH=''

testFile=''#FLAGS.ply_path
#print(onlyPlyfiles)
reverseDict=dict({0:"bird",1:"can",2:"cracker",3:"house",4:"shoe"})
NUM_CLASSES = 5

import pointnet_colored as MODEL_onlycolored
import onevsall as MODEL_onlyPoints
import pointnet_colored as MODEL_onlynormals
import pointnet_coloredNormals as MODEL_normalsandcolors

pipelineCode=3  
'''
0 = colored
1 = colored+normals
2 = only points
3 = only normals
'''

MODEL=None

if pipelineCode ==0:
	MODEL = MODEL_onlycolored
	MODEL_PATH='logcolored/model.ckpt'
elif pipelineCode ==1:
	MODEL = MODEL_normalsandcolors
	MODEL_PATH='logNORMALSncolored/model.ckpt'
elif pipelineCode==2:
	MODEL = MODEL_onlyPoints
	MODEL_PATH = 'log/model.ckpt'
elif pipelineCode==3:
	MODEL = MODEL_onlycolored
	MODEL_PATH='logdatanormalsnocolor/model.ckpt'










is_training = False
pointclouds_pl=None
pointclouds_rgb_pl=None
labels_pl=None
is_training_pl = tf.placeholder(tf.bool, shape=())

if pipelineCode==2:
	pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
	pred, end_points = MODEL.get_model(pointclouds_pl,is_training_pl)
else:# pipelineCode ==0:
	pointclouds_pl,pointclouds_rgb_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
	pred, end_points = MODEL.get_model(pointclouds_pl, pointclouds_rgb_pl,is_training_pl)

	
#loss = MODEL.get_loss(pred, labels_pl, end_points)
pred = tf.nn.softmax(pred)
# Add ops to save and restore all the variables.
saver = tf.train.Saver()
    
# Create a session


sess = tf.Session()

# Restore variables from disk.
saver.restore(sess, MODEL_PATH)
#log_string("Model restored.")
ops={'pointclouds_pl': pointclouds_pl,
       'is_training_pl': is_training_pl,
       'pred': pred,
       }
if pipelineCode!=2:
	ops['pointclouds_rgb_pl']=pointclouds_rgb_pl



path2colorsavgSigma='colorsnormalstrain.h5trainAverageStdColor.txt'





from io import StringIO

class StringBuilder:
     _file_str = None

     def __init__(self):
         self._file_str = StringIO()

     def Append(self, str):
         self._file_str.write(str)

     def __str__(self):
         return self._file_str.getvalue()



def evaluate(num_votes):

    eval_one_epoch(sess, ops, num_votes)

   
def eval_one_epoch(testFile=testFile,sess=sess,ops=ops,num_votes=1):
    is_training = False
    current_data=[]

    current_data,current_colors,current_normals = provider.load_ply_data(testFile,NUM_POINT,path2colorsavgSigma)
    #current_label = np.squeeze(current_label)
    if pipelineCode==1:
    	current_colors = np.concatenate((current_colors,current_normals),axis=1)
    if pipelineCode==3:
    	current_colors = current_normals
    current_data = np.asarray([current_data,np.zeros_like(current_data)])
    current_colors = np.asarray([current_colors,np.zeros_like(current_colors)])
    current_normals = np.asarray([current_normals,np.zeros_like(current_normals)])
    #print(current_data.shape)
            
    #file_size = current_data.shape[0]
    num_batches = 1
    #print(file_size)
      
    
    batch_pred_sum = np.zeros((current_data.shape[0], NUM_CLASSES)) # score for classes
    batch_pred_classes = np.zeros((current_data.shape[0], NUM_CLASSES)) # 0/1 for classes
    feed_dict = {ops['pointclouds_pl']: current_data,
                 
                 ops['is_training_pl']: is_training}
    if pipelineCode!=2:
    	feed_dict[ops['pointclouds_rgb_pl']]=current_colors
    pred_val = sess.run( ops['pred'],feed_dict=feed_dict)
    sb = StringBuilder()
    #if(len(onlyPlyfiles)==0):
    #    onlyPlyfiles.append(testFile)
    #for i in range(len(onlyPlyfiles)):
    sb.Append(str(np.max(pred_val[0])))
    sb.Append(",")
    sb.Append(reverseDict[np.argmax(pred_val[0])])
    #sb.Append('\n')
            #print(str(np.max(pred_val[i]))+","+reverseDict[np.argmax(pred_val[i])])

    return sb.__str__();
app = Flask(__name__)



@app.route('/', methods=['POST'])
def main():
    #input = ((255 - np.array(request.json, dtype=np.uint8)) / 255.0).reshape(1, 784)
    #testFile = request.args.get('testFile')
    testFile = request.data
    output = eval_one_epoch(testFile=testFile)
    return output


if __name__=='__main__':
    with tf.device('/cpu:0'):
        with tf.Graph().as_default():
                app.run()
