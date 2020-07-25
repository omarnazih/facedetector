import cv2
from keras.preprocessing import image
# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Read the input image
img = cv2.imread('9.jpeg')
# Convert into grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.1, 4)
# Draw rectangle around the faces
for (x, y, w, h) in faces:
    crop = img[y:y+h, x:x+w]
# Display the output
cv2.imshow('img', crop)
cv2.waitKey()



gray2 = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)


x = image.img_to_array(gray2)
x = np.expand_dims(x, axis = 0)

x /= 255

x = np.array(x, 'float32')
x = x.reshape([48, 48]);

print(x)


