import numpy as np
import cv2 as cv
import os

def dist(p1,p2,p):
  # p = 2 for eucledian distance
  return np.sqrt(((p1-p2)**p).sum())

def knn(train,test,k=5):
  print('Train :',train.shape)
  print('Test :',test.shape)
  distances = []
  for i in range(train.shape[0]):
    x = train[i, :-1]
    y = train[i,-1]
    d = dist(test,x,2)
    distances.append([d,y])
  print("Distances = ",distances)
  K_distances = sorted(distances, key=lambda x: x[0])[:k] # closest K distances
  print("K_distances = ",K_distances)
  labels = np.array(K_distances)[:,-1]
  print("labels = ",labels)
  output = np.unique(labels,return_counts=True)
  print("output = ",output)
  index = np.argmax(output[1])
  print("index = ",index)
  print(output[0][index])
  return output[0][index]

cap = cv.VideoCapture(0)
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')

data_Path = './face_data/'
face_data = []
labels = []
class_ID = 0 #labels
names = {}

for file in os.listdir(data_Path):
  if file.endswith('.npy'):
    names[class_ID] = file.split('#')[0]
    item = np.load(data_Path+file)
    face_data.append(item)
    target = class_ID * np.ones((item.shape[0],))
    class_ID += 1
    labels.append(target)

print('Names ::',names)

face_data = np.concatenate(face_data, axis=0)
print('Shape FD:',face_data.shape)
face_labels = np.concatenate(labels, axis=0).reshape((-1,1))
print('Shape FL:',face_labels.shape)

trainSet = np.concatenate((face_data,face_labels), axis = 1)
print('Training set :: ',trainSet.shape)

while cap.isOpened():
  ret, frame = cap.read()
  
  if ret:
    
    g_frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(g_frame,scaleFactor=1.05, minNeighbors=5, minSize=(30, 30),flags=cv.CASCADE_SCALE_IMAGE)
    
    if len(faces) != 0:
      for face in faces:  
        # get region of interest
        x,y,w,h = face
        padding = 5
        try : 
          faceROI = frame[y-padding:y+h+padding, x-padding:x+w+padding]
          face_section = cv.resize(faceROI,(100,100))
        except Exception as e:
                    print(str(e))
        # predict
        out = knn(trainSet,face_section.flatten())

        # mark it
        cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
        cv.putText(frame,names[int(out)],(x,y-10),cv.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1,cv.LINE_AA)
    
    cv.imshow('frame',frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
      break

cap.release()
cv.destroyAllWindows()

