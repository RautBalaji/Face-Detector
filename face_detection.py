from mtcnn import MTCNN
import cv2
import matplotlib.pyplot as plt



cap = cv2.VideoCapture('D:\\assignments\\face_detection\\running.mp4')
detector = MTCNN()
#img = cv2.imread('/content/drive/MyDrive/Colab_Notebooks/deep_learning/face detection/face_detect3.jpg')

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame,(600,500))
    output = detector.detect_faces(frame)
    for i in output:

        x,y,w,h = i['box']
        #left_eyeX,left_eyeY = i['keypoints']['left_eye']
        #right_eyeX,right_eyeY = i['keypoints']['right_eye']
        #nose_X,nose_Y = i['keypoints']['nose']
        #mouth_leftX,mouth_leftY = i['keypoints']['mouth_left']
        #mouth_rightX,mouth_rightY = i['keypoints']['mouth_right']

        cv2.rectangle(frame,pt1=(x,y),pt2=(x+w,y+h),color=(255,255,0),thickness=3)
    cv2.imshow("win",frame)
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break



cv2.destroyAllWindows()    