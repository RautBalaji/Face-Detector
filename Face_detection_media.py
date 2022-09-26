import enum
from charset_normalizer import detect
import cv2

import mediapipe as mp
import time
#cap = cv2.VideoCapture('D:\\OpenCV\Adavanced_Computer_Vision\\data\\running.mp4')
cap = cv2.VideoCapture(0)
cTime = 0
pTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(0.75)#intialize face detecton

while True:
    success,img = cap.read()
    img =cv2.resize(img,(800,500))
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)

    if results.detections:
        for id,detection in enumerate(results.detections):
            #mpDraw.draw_detection(img,detection)
            print(id,detection)
            #print(detection.score)
            #print(detection.location_data.relative_bounding_box)
            bboxC = detection.location_data.relative_bounding_box
            h,w,c = img.shape
            #get the xmin,ymin,width,height value for our own rectangle around face
            bbox =int(bboxC.xmin * w),int(bboxC.ymin * h), int(bboxC.width * w),int(bboxC.height * h)
                  
                  
            cv2.rectangle(img,bbox,(255,0,255),2) 
            cv2.putText(img,f'{int(detection.score[0]*100)}%',(bbox[0],bbox[1]-20),
                        cv2.FONT_HERSHEY_DUPLEX,1,(0,255,0),2)
     
    

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img,"FPS:"+str(int(fps)),(28,78),cv2.FONT_HERSHEY_DUPLEX,1,(255,0,0),2)

    cv2.imshow("image",img)
    cv2.waitKey(20)