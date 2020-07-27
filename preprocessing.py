import numpy as np
import cv2
#from keras.preprocessing.image import img_to_array
#from keras.models import load_model

#The image used for test
img_path = 'img/5.jpeg'

#Detection model path
haarcascade_path = 'Detection model/haarcascade_frontalface_default.xml'
emotionmodel_path= 'Emotion model/_mini_XCEPTION.102-0.66.hdf5'

#loading emotion model
#model = load_model(emotionmodel_path,compile=False)

#loading the facedetection model
face_cascade = cv2.CascadeClassifier(haarcascade_path)

#Emotions
EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised","neutral"]


img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
    
#for (x, y, w, h) in faces:
 #   roi = gray[y:y+h, x:x+w]



    if len(faces) > 0:
        faces = sorted(faces, reverse=True,
        key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces
        # Extract the ROI of the face from the grayscale image, resize it to a fixed 28x28 pixels, and then prepare
        # the ROI for classification via the CNN
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (64, 64))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        
        
        preds = emotion_classifier.predict(roi)[0]
        emotion_probability = np.max(preds)
        label = EMOTIONS[preds.argmax()]
    else: continue


cv2.imshow('gray',roi)
cv2.waitKey(0)
cv2.destroyAllWindows()



