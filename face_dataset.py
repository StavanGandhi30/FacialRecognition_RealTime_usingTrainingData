import cv2

#Video Frame
cam = cv2.VideoCapture(0)
cam.set(0, 1000)
# cam.set(4, 480)

#XML haarcascade_frontalface_default File
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Number of Images to capture.
num_of_sample = 100
count = 0
while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        count += 1
        cv2.imwrite("dataset/User." + str(1) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
        cv2.imshow('image', img)

    if cv2.waitKey(100) & 0xff == 27:   # Press 'ESC' for exiting video
        break
    elif count >= num_of_sample:  #Number of Images to capture.
        break

cam.release()
cv2.destroyAllWindows()
