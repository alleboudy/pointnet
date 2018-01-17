# El detectozord!
![El Detectozord](https://github.com/alleboudy/pointnet/blob/master/doc/eldetectozord.png?raw=true "El detectoZord")

### please see the updates down at the end for some clarifications [Updates based on discussions section]

in this repository I editted PointNet to use colors and normals as well

added in utils/ scripts to prepare training data from .ply files that are centered at the origian and in a unit bounding box

classify.py for classifying raw .ply files of objects [it does the centeralization and unit bounding box in provider.py already]

onlineclassify.py a simple flask app to serve the classification on a web server with the ability to set the used variety of poinetNet
in it where 

pipelineCode=
'''
0 = colored
1 = colored+normals
2 = only points
3 = only normals
'''

please see requestclassification.py for a simple client that uses the server

regress.py uses a different variation of PoinetNet can be found under models/leboudyNet.py that tries to regress the pose of a given object, however it needs better training data


in https://github.com/alleboudy/detectozord
a typical use in a detection pipeline is present in /segmentation

new updates are to be added later ...
Thanks!
-------

# Updates based on discussions:


## As per the normalization part,
For python: in 
https://github.com/alleboudy/pointnet/blob/master/provider.py
after line 211 


minx = min(sampled_pcxyz_array[:,0])
miny = min(sampled_pcxyz_array[:,1])
minz = min(sampled_pcxyz_array[:,2])
maxx = max(sampled_pcxyz_array[:,0])
maxy = max(sampled_pcxyz_array[:,1])
maxz = max(sampled_pcxyz_array[:,2])
scale = min((1 / (maxx - minx)), min(1 / (maxy - miny),1/ (maxz-minz)))
sampled_pcxyz_array[:,0] = (sampled_pcxyz_array[:,0] - 0.5*(minx + maxx))*scale + 0.5
sampled_pcxyz_array[:,1] = (sampled_pcxyz_array[:,1] - 0.5*(miny + maxy))*scale + 0.5
sampled_pcxyz_array[:,2] = (sampled_pcxyz_array[:,2] - 0.5*(minz + maxz))*scale + 0.5
sampled_pcxyz_array[:,0] -= np.average(sampled_pcxyz_array[:,0])
sampled_pcxyz_array[:,1] -= np.average(sampled_pcxyz_array[:,1])
sampled_pcxyz_array[:,2] -= np.average(sampled_pcxyz_array[:,2])
could have definitely been written in a cleaner way, but uh, desperate times call for desperate measures xD

attached is the snippit for C++ as well, 
it is in the repository 
https://github.com/alleboudy/detectozord

https://github.com/alleboudy/detectozord/blob/master/utils/utils/main.cpp

from line 240 onwards, 

I would usually use the c++ for when processing the training data, and the python for when doing the classification, but they are exactly the same[notice the 'scale' variable to respect the aspect ratio, without it, you get very funny shapes 😃]


------------
## Training Different Models:

### For normals only; I used the regular colored architecture but fed it normals instead of colors, I know I should have changed its name in a separate file, sorry xD

https://github.com/alleboudy/pointnet/blob/master/models/pointnet_colored.py


### For normals and colors;

https://github.com/alleboudy/pointnet/blob/master/models/pointnet_coloredNormals.py

------
### Please, feel free to contact me for any clarifications, also, if you find this useful in a research you do, I would really appreciate it so much if you may reference my work here ^^
Ahmad Alleboudy, ahmad.alleboudy@outlook.com,
*M.Sc. student of the Computer Science department at Pisa University, Italy*

https://www.linkedin.com/in/ahmadalleboudy/
