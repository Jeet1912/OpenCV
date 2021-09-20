import cv2 as cv

cap = cv.VideoCapture(0)
cap.set(3,600)
cap.set(4,500)

classNames = []
classFile = 'coco.names'
with open(classFile,'r') as f:
   classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weights = 'frozen_inference_graph.pb'
net = cv.dnn_DetectionModel(weights, configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)

while True:
  ret, frame = cap.read()
  if ret:
    classIds, confs, bbox = net.detect(frame,confThreshold=0.6)

    if len(classIds) != 0:
      for id, confidence, bb in zip(classIds.flatten(), confs.flatten(),bbox):
        if id > 80:
          #limited by coco.names
          continue
        cv.rectangle(frame,bb,color=(0,255,0),thickness=1)
        cv.putText(frame,classNames[id-1].upper(),(bb[0]+10,bb[0]+30),cv.FONT_HERSHEY_COMPLEX,1,(0,200,0),2)
        cv.imshow('frame',frame)
  else:
    break
  if cv.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv.destroyAllWindows()
for i in range (1,5):
    cv.waitKey(1)
    


