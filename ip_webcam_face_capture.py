'''
    Facial Recognition Side Project

    File: ip_webcam_face_capture.py
    Date: 11/9/17
    Author: Alfredo Salazar
    Version: 1.0

    A program that takes frames from IP Webcam (a website that uses my android
    phone as a webcam), detects faces and eyes using Haar cascades, and displays video
    and the detections in a window.
'''

import numpy as np
import cv2, urllib, time

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')

# URL to get frames of IP Webcam
url = raw_input("\nEnter IPv4 shown on IP Webcam app: ") + "/shot.jpg"

cv2.namedWindow('Webcam Face Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Webcam Face Detection', 720, 640)

while True:
     # Use urllib to get a frame from the webcam
    imgResp = urllib.urlopen(url)
    
    # Convert frame into a Numpy array
    imgNp = np.array(bytearray(imgResp.read()),dtype=np.uint8)
    
    # Decode the array to OpenCV usable format
    img = cv2.imdecode(imgNp,-1)

    # Check if there's no frame
    if img is None:
        break

    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect multiple sized faces in gray image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        # Draw blue rectangles where faces are detected
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        #'''
        # Define regions of image for eyes
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        # Detect multiple sized eyes in gray ROI
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        for (ex,ey,ew,eh) in eyes:
            # Draw green rectangles where eyes are detected
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        #'''

    # Show color key
    cv2.putText(img, "Blue is face. Green is eyes.", (0, 30), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 255))
    # Display frame/image to window
    cv2.imshow('Webcam Face Detection',img)

    # Wait 1 ms for keypress
    k = cv2.waitKey(1)

    # Check if escape key is pressed
    if k == 27:
        break
    
cv2.destroyAllWindows()
