import cv2
import face_recognition

image = face_recognition.load_image_file('images.jpg')

cv2.imshow('hewan', image)

cv2.waitKey(0)
cv2.destroyWindow()