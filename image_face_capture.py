import numpy as np
import cv2, urllib, time, os, sys, os.path

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')
cat_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalcatface.xml')
stop_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_stopsign.xml')

print "\nAvailable image filenames: "
files = os.listdir("images")
newlist = []
for names in files:
    if names.endswith(".jpg") or names.endswith(".png"):
        newlist.append(names)
print newlist

# Ask for name of image
print ""
imgname_entry_valid = False
while imgname_entry_valid == False:
    imgname = raw_input("Filename of image? ")

    if os.path.isfile("images/" + imgname):
        imgname_entry_valid = True
    else:
        print "Image filename was incorrect, try again.\n"
print ""

img = cv2.imread("images/" + imgname)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

cats = cat_cascade.detectMultiScale(gray, 1.7, 5)
for (cx,cy,cw,ch) in cats:
    cv2.rectangle(img,(cx,cy),(cx+cw,cy+ch),(0,0,255),2)

stops = stop_cascade.detectMultiScale(gray, 1.3, 5)
for (sx,sy,sw,sh) in stops:
    cv2.rectangle(img,(sx,sy),(sx+sw,sy+sh),(0,128,255),2)

length = np.size(img, 0)
cv2.putText(img, "Blue=face Green=eyes Red=cat Orange=stop", (0, length-10), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 255))

cv2.namedWindow("Facial Detection", cv2.WINDOW_NORMAL)        
cv2.imshow("Facial Detection", img)
cv2.resizeWindow("Facial Detection", 1280, 720)
cv2.waitKey(0)
cv2.destroyAllWindows()
