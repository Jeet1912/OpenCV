# testing it on an image

import cv2 as cv

img = cv.imread('/Users/jxxt/OpenCV/object_detection/lena.png')

#loading classes
classNames = []
classFile = 'coco.names'
with open(classFile,'r') as f:
   classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weights = 'frozen_inference_graph.pb'
net = cv.dnn_DetectionModel(weights, configPath)
#source == documentation
net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)

classIds, confs, bbox = net.detect(img,confThreshold=0.5)

for id, confidence, bb in zip(classIds.flatten(), confs.flatten(),bbox):
  print(bb)
  cv.rectangle(img,bb,color=(0,255,0),thickness=1)
  cv.putText(img,classNames[id-1].upper(),(bb[0]+10,bb[0]+30),cv.FONT_HERSHEY_COMPLEX,1,(0,200,0),2)



cv.imshow('img',img)
cv.waitKey(0)