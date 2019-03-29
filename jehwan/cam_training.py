import os
import cv2
import random
name = input("이름:")

cam = cv2.VideoCapture(0)
if cam.isOpened() == False:
    print('unable to read camera feed')
count = 0
if not os.path.isdir('./image/' + name):
    os.mkdir('./image/'+ name)

while(True):
    ret, img = cam.read()
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if ret:
        cv2.imshow('image', img)
        cv2.imwrite("./image/"+name+'/'+ str(random.randint(1, 100000)) + ".jpg", img)
        count += 1
    k = cv2.waitKey(1) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 30: # Take 30 face sample and stop video
         break

cam.release()
cv2.destroyAllWindows()