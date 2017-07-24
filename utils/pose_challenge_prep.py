from plyfile import (PlyData, PlyElement, make2d, PlyParseError, PlyProperty)
from random import shuffle
import numpy as np
import h5py

from os import listdir
from os.path import isfile, join

modelname='shoe'
mainplyDir='trainData/'+modelname+'/clouds'
plyfiles2load=[f for f in listdir(mainplyDir) if isfile(join(mainplyDir, f))]
#['bird-.ply','bond-.ply','can-.ply','cracker-.ply','shoe-.ply','teapot-.ply']
outputh5FilePath=modelname+'posestrain.h5'



# Write numpy array data and label to h5_filename
def save_h5_data_label_normal(h5_filename, data, posesx, posesq,labels, 
        data_dtype='float32', label_dtype='uint8'):
    h5_fout = h5py.File(h5_filename)
    h5_fout.create_dataset(
            'data', data=data,
            compression='gzip', compression_opts=4,
            dtype=data_dtype)
    
    h5_fout.create_dataset(
            'posesx', data=posesx,
            compression='gzip', compression_opts=1,
            dtype=data_dtype)
    h5_fout.create_dataset(
            'posesq', data=posesq,
            compression='gzip', compression_opts=1,
            dtype=data_dtype)
    h5_fout.create_dataset(
            'labels', data=labels,
            compression='gzip', compression_opts=1,
            dtype=label_dtype)
    h5_fout.close()
# Load PLY file
def load_ply_data(filename):
    try:
        
        plydata = PlyData.read(filename)
        pc = plydata['vertex'].data
        pcxyz_array=[]
        pcnxyz_array=[]
        sampled_pcxyz_array=[]
        sampled_pcnxyz_array=[]
        for w in pc:
            x=w[0]
            y=w[1]
            z=w[2]
            pcxyz_array.append([x, y, z])
            #pcnxyz_array.append([_nx,_ny,_nz])
        indices = list(range(len(pcxyz_array)))
        indicessampled= np.random.choice(indices, size=2048)
        for i in indicessampled:
            sampled_pcxyz_array.append(pcxyz_array[i])
           # sampled_pcnxyz_array.append(pcnxyz_array[i])

        return np.asarray(sampled_pcxyz_array),np.zeros_like(sampled_pcxyz_array)
    except :
        print('err loading file')


#TODO: load the quaternions and translation for the label and use it to build the dataset!!
def load_poses_ply_data(filename):
    try:
        
        plydata = PlyData.read(filename)
        pc = plydata['vertex'].data
        pcxyz_array=[]
        sampled_pcxyz_array=[]
        posesx=[]
        posesq=[]
        pose=""
        with open(filename.replace('clouds','poses').replace('.ply','.txt')) as fp:
            pose = fp.read()
        pose = pose.split(',')
        posesx=[float(c) for c in pose[:3]]
        #print(posesx)
        posesq=[float(c) for c in pose[3:]]
       
        for w in pc:
            x=w[0]
            y=w[1]
            z=w[2]
            pcxyz_array.append([x, y, z])
        indices = list(range(len(pcxyz_array)))
        indicessampled= np.random.choice(indices, size=2048)
        for i in indicessampled:
            sampled_pcxyz_array.append(pcxyz_array[i])
        return np.asarray(sampled_pcxyz_array),np.asarray(posesx),np.asarray(posesq)
    except :
        print('err loading file')




labelsMap = dict({"bird":0,"can":1,"cracker":2,"house":3,"shoe":4})

allpoints=[]
posesx=[]
posesq=[]
alllabels=[]
counter=0
for plyFile in plyfiles2load:
    #print(plyFile)
    counter+=1
    print("file number: "+str(counter))
    try:
        plyxyz,posex,poseq = load_poses_ply_data(join(mainplyDir,plyFile))
        allpoints.append(plyxyz)
        #print(posex)
        posesx.append(posex)
        posesq.append(poseq)
        alllabels.append(np.asarray([labelsMap[plyFile.split('-')[0]]]))
    except:
        print('err loading file')
        continue


indices=list(range(len(allpoints)))
shuffle(indices)

allpoints_shuffle = [allpoints[i] for i in indices] 
posesx_shuffle = [posesx[i] for i in indices] 
posesq_shuffle = [posesq[i] for i in indices] 
alllabels_shuffle = [alllabels[i] for i in indices] 

#print(posesx_shuffle)


save_h5_data_label_normal(outputh5FilePath,
    np.asarray(allpoints_shuffle),np.asarray(posesx_shuffle),np.asarray(posesq_shuffle),
    np.asarray(alllabels_shuffle))


