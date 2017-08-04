# El detectozord!
![El Detectozord](https://github.com/alleboudy/pointnet/blob/master/doc/eldetectozord.png?raw=true "El detectoZord")

a comprehensive readme will be presented later, but a few words for now .... and apologies for the experimental code, will clean it up later

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

new updates are to be added later ...
Thanks!

