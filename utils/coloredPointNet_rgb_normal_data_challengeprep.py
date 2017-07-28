from plyfile import (PlyData, PlyElement, make2d, PlyParseError, PlyProperty)
from random import shuffle
import numpy as np
import h5py
from os import listdir
from os.path import isfile, join

#modelname='shoe'

mainplyDir='trainplyfiles'
plyfiles2load=[f for f in listdir(mainplyDir) if isfile(join(mainplyDir, f))]
#['bird-.ply','bond-.ply','can-.ply','cracker-.ply','shoe-.ply','teapot-.ply']
outputh5FilePath='colorsnormalstrain.h5'
avgstdfile=outputh5FilePath+'trainAverageStdColor.txt'


# Write numpy array data and label to h5_filename
def save_h5_data_label_normal(h5_filename, data, normals, rgb,labels, 
        data_dtype='float32', label_dtype='uint8'):
    h5_fout = h5py.File(h5_filename)
    h5_fout.create_dataset(
            'data', data=data,
            compression='gzip', compression_opts=1,
            dtype=data_dtype)
    
    h5_fout.create_dataset(
            'normals', data=normals,
            compression='gzip', compression_opts=1,
            dtype=data_dtype)
    h5_fout.create_dataset(
            'rgb', data=rgb,
            compression='gzip', compression_opts=1,
            dtype=data_dtype)
    h5_fout.create_dataset(
            'labels', data=labels,
            compression='gzip', compression_opts=1,
            dtype=label_dtype)
    h5_fout.close()


#TODO: load the quaternions and translation for the label and use it to build the dataset!!
def load_poses_ply_data(filename):
   # try:
        
        plydata = PlyData.read(filename)
        pc = plydata['vertex'].data
        pcxyz_array=[]
        sampled_pcxyz_array=[]
        normals=[]
        rgb=[]
        sampled_rgb=[]
        sampled_normals=[]
        #pose=""
        #with open(filename.replace('clouds','poses').replace('.ply','.txt')) as fp:
        #    pose = fp.read()
        #pose = pose.split(',')
        #normals=[float(c) for c in pose[:3]]
        #print(normals)
        #rgb=[float(c) for c in pose[3:]]
       
        for s in pc:
            w=list(s)
            #print(s)
            pcxyz_array.append(w[0:3])
            normals.append(w[3:6])
            rgb.append(w[6:9])
        indices = list(range(len(pcxyz_array)))
        indicessampled= np.random.choice(indices, size=2048)
        for i in indicessampled:
            sampled_pcxyz_array.append(pcxyz_array[i])
            sampled_rgb.append(rgb[i])
            sampled_normals.append(normals[i])
        return np.asarray(sampled_pcxyz_array),np.asarray(sampled_normals),np.asarray(sampled_rgb)
    #except :
     #   print('err loading file')




labelsMap = dict({"bird":0,"can":1,"cracker":2,"house":3,"shoe":4})

allpoints=[]
normals=[]
rgb=[]
alllabels=[]
counter=0
for plyFile in plyfiles2load:
    #print(plyFile)
    counter+=1
    print("file number: "+str(counter))
    try:
        plyxyz,normal,color = load_poses_ply_data(join(mainplyDir,plyFile))
        allpoints.append(plyxyz)
        #print(posex)
        normals.append(normal)
        rgb.append(color)
        alllabels.append(np.asarray([labelsMap[plyFile.split('-')[0]]]))
    except:
        print('err loading file')
        continue


indices=list(range(len(allpoints)))
shuffle(indices)

allpoints_shuffle = [allpoints[i] for i in indices] 
normals_shuffle = [normals[i] for i in indices] 
rgb_shuffle = [rgb[i] for i in indices] 
alllabels_shuffle = [alllabels[i] for i in indices] 

#print(normals_shuffle)
standardDiv = np.std(rgb_shuffle)
print('std:',standardDiv)
averageColor=np.average(rgb_shuffle)
print('average color:',averageColor)
rgb_shuffle-=averageColor
rgb_shuffle/=standardDiv

with open( avgstdfile, 'w')as myfile:
            myfile.write(str(averageColor)+','+str(standardDiv))
save_h5_data_label_normal(outputh5FilePath,
    np.asarray(allpoints_shuffle),np.asarray(normals_shuffle),np.asarray(rgb_shuffle),
    np.asarray(alllabels_shuffle))


