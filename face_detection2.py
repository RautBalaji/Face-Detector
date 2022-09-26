from mtcnn import MTCNN
import cv2
import matplotlib.pyplot as plt

detector = MTCNN()
img = cv2.imread('D:\\assignments\\face_detection\\tilt_face.jpg')
img = cv2.resize(img,(1000,700))

output = detector.detect_faces(img)
for i in output:

  x,y,w,h = i['box']
  left_eyeX,left_eyeY = i['keypoints']['left_eye']
  right_eyeX,right_eyeY = i['keypoints']['right_eye']
  nose_X,nose_Y = i['keypoints']['nose']
  mouth_leftX,mouth_leftY = i['keypoints']['mouth_left']
  mouth_rightX,mouth_rightY = i['keypoints']['mouth_right']

  cv2.rectangle(img,pt1=(x,y),pt2=(x+w,y+h),color=(255,255,0),thickness=3)
  cv2.circle(img,center=(left_eyeX,left_eyeY),color=(255,255,0),thickness=3,radius=5)
  cv2.circle(img,center=(right_eyeX,right_eyeY),color=(255,255,0),thickness=3,radius=5)
  cv2.circle(img,center=(nose_X,nose_Y),color=(255,255,0),thickness=3,radius=5)
  cv2.circle(img,center=(mouth_leftX,mouth_leftY),color=(255,255,0),thickness=3,radius=5)
  cv2.circle(img,center=(mouth_rightX,mouth_rightY),color=(255,255,0),thickness=3,radius=5)

cv2.imshow("face_detection",img)
cv2.waitKey(0)

cv2.destroyAllWindows() 