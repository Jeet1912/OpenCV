import cv2 as cv
import numpy as np
import datetime as dt

cap = cv.VideoCapture(0)
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')
skip = 0
face_data = []
data_path = './face_data/'
filename = input('Enter the name of the person: ')
filename = filename + '#' + str(dt.datetime.now().timestamp())

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        g_frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        # g_frame,1.3,5
        faces = face_cascade.detectMultiScale(g_frame,scaleFactor=1.05, minNeighbors=5, minSize=(30, 30),flags=cv.CASCADE_SCALE_IMAGE)
        # faces - coordinates (x,y,w,h)
        if len(faces) != 0:
            # sorting faces in descending order of area
            faces = sorted(faces, key = lambda x: x[2] * x[3], reverse=True)
            skip += 1
            k = 1
    
            for face in faces:
                x,y,w,h = face
                padding = 5 
                try : 
                    face_offset = frame[y-padding:y+h+padding, x-padding:x+w+padding]
                    face_selection = cv.resize(face_offset,(100,100))
                except Exception as e:
                    print(str(e))
                if skip % 10 == 0:
                    # store in 10 seconds
                    face_data.append(face_selection)

                cv.imshow(str(k),face_selection)
                k += 1
                cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
        
        cv.imshow('frame',frame)    
        if cv.waitKey(1) & 0xFF == ord('q'):
            break


face_data = np.array(face_data)
#print('shape ',face_data.shape)
face_data = face_data.reshape((face_data.shape[0],-1))

np.save(data_path+filename,face_data)
#print('Data saved at :{}'.format(data_path+filename+'.npy'))
cap.release()
cv.destroyAllWindows()

"""
Observations - 
  Haar classifier is fairly accurate, however it fails to recognize the facial features when the person is not directly looking at the camera.
"""