import cv2

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')   #load trained model
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, img = cv2.VideoCapture(0).read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 5)
    for(x,y,w,h) in faces:
        _, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        if (confidence < 100):
            confidence = "{0}".format(round(100 - confidence))
            if int(confidence)>=70:
                print('Logged In!!')
                exit()
            else:
                print(confidence)