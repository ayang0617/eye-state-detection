#coding=utf-8
import time
import imutils
import numpy as np
import cv2
import dlib
from scipy.spatial import distance
import os
from imutils import face_utils
from sklearn import svm
from sklearn.externals import joblib

VECTOR_SIZE = 3
def queue_in(queue, data):
    ret = None
    if len(queue) >= VECTOR_SIZE:
        ret = queue.pop(0)
    queue.append(data)
    return ret, queue

def eye_aspect_ratio(eye):
    # print(eye)
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

pwd = os.getcwd()
model_path = os.path.join(pwd, 'model')
shape_detector_path = os.path.join(model_path, '/Users/ayang/PycharmProjects/pythonProject/shape_predictor_68_face_landmarks.dat')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_detector_path)

# 导入模型
clf = joblib.load("/Users/ayang/PycharmProjects/pythonProject/bestmodel.pth")

EYE_AR_THRESH = 0.25 # EAR阈值
EYE_AR_CONSEC_FRAMES = 2# 当EAR小于阈值时，接连多少帧一定发生眨眼动作

# 对应特征点的序号
RIGHT_EYE_START = 37 - 1
RIGHT_EYE_END = 42 - 1
LEFT_EYE_START = 43 - 1
LEFT_EYE_END = 48 - 1
COUNTER = 0
TOTAL = 0
ear_vector = []
alarm = False
cap = cv2.VideoCapture(0)
start = time.time()
while True:

    ret, frame = cap.read()
    frame = imutils.resize(frame, width=720)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 第六步：使用detector(gray, 0) 进行脸部位置检测
    rects = detector(gray, 0)


    def cos_angle(a, b, c):
        x, y = b - a, b - c
        Lx = np.sqrt(x.dot(x))  # a.dot(b) 与 np.dot(a,b)效果相同  , a b 均为矩阵
        Ly = np.sqrt(y.dot(y))
        cos_angle = x.dot(y) / (Lx * Ly)  # 角的临边比斜边的值
        # 根据条件选择是计算弧度还是角度
        return np.arccos(cos_angle) * 360 / 2 / np.pi  # 结果是反余弦函数


    def normalization(path):

        detector = dlib.get_frontal_face_detector()
        landmark_predictor = dlib.shape_predictor(
            r'/Users/ayang/PycharmProjects/pythonProject/shape_predictor_68_face_landmarks.dat')
        img = cv2.imread(path)
        faces = detector(img, 1)
        feas = []  # 关键点

        top, bottom, left, right = 0, 0, 0, 0

        if (len(faces) > 0):
            for k, d in enumerate(faces):  # enenumerate表示枚举  显示格式是 编号,元素 比如:0 one

                bottom = d.bottom()
                top = d.top()
                left = d.left()
                right = d.right()

                cv2.rectangle(img, (left, top), (right, bottom), (255, 255, 255))
                shape = landmark_predictor(img, d)

                for i in range(68):
                    num = str(shape.part(i))[1:-1].split(",")
                    feas.append([int(num[0]), int(num[1])])

        feas = np.array(feas)
        s_fa = feas[45, :][1] - feas[36, :][1]
        d = feas[45, :][0]
        e = feas[36, :][1]

        a, b, c = feas[45, :], feas[36, :], np.array([d, e])
        angle = cos_angle(a, b, c)

        if s_fa < 0:
            angle = 0 - angle

        # if abs(s_fa) > 5:
        #     if s_fa > 0 and angle >35:
        #         return angle
        #     elif s_fa < 0 and angle >35:
        #         angle = 360 - angle
        #         return angle
        #     else:
        #         return 0
        # else:
        #     return 0

        # 先旋转在截图,因为先截图的话,没办法保证人脸是全的

        dst = img[top:bottom, left:right]

        matRotate = cv2.getRotationMatrix2D(((right - left) / 2, (bottom - top) / 2), angle, 1)  # 变换矩阵
        rot_img = cv2.warpAffine(dst, matRotate, ((right - left), (bottom - top)))
        # cv2.imshow("rot_img",rot_img)
        # cv2.waitKey(0)

        res_dst = cv2.resize(rot_img, (50, 50))

        cv2.imshow("res_dst", rot_img)
        cv2.waitKey(0)


    # 第七步：循环脸部位置信息，使用predictor(gray, rect)获得脸部特征位置的信息
    for rect in rects:
        print('-' * 20)
        shape = predictor(gray, rect)

        # 第八步：将脸部特征信息转换为数组array的格式
        shape = face_utils.shape_to_np(shape)
        # 第九步：提取左眼和右眼坐标
        leftEye = shape[LEFT_EYE_START:LEFT_EYE_END + 1]
        rightEye = shape[RIGHT_EYE_START:RIGHT_EYE_END + 1]

        # 第十步：构造函数计算左右眼的EAR值，使用平均值作为最终的EAR
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        if ear < EYE_AR_THRESH:
            now = 'closed'
        else:
            now = 'opened'
       # print('leftEAR = {0}'.format(leftEAR))
       # print('rightEAR = {0}'.format(rightEAR))
        print('now = {0}'.format(now))

        # 第十一步：使用cv2.convexHull获得凸包位置，使用drawContours画出轮廓位置进行画图操作
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # 第十二步：进行画图操作，用矩形框标注人脸
        ret, ear_vector = queue_in(ear_vector, ear)
        """
        if (len(ear_vector) == VECTOR_SIZE):
            print(ear_vector)
            input_vector = []
            input_vector.append(ear_vector)
            res = clf.predict(input_vector)
            print(res)
        """

        ret, ear_vector = queue_in(ear_vector, ear)
        left = rect.left()
        top = rect.top()
        right = rect.right()
        bottom = rect.bottom()
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)


        '''
            分别计算左眼和右眼的评分求平均作为最终的评分，如果小于阈值，则加1，如果连续3次都小于阈值，则表示进行了一次眨眼活动
        '''
        # 第十三步：循环，满足条件的，眨眼次数+1

        if ear < EYE_AR_THRESH:  # 眼睛长宽比：0.2
            COUNTER += 1
            if COUNTER ==15:
                if not alarm:
                    alarm = True
                    cv2.putText(frame, "wake up!!!", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            if COUNTER > 15:
                for i in range(0, 2):
                    #os.system('afplay /System/Library/Sounds/Sosumi.aiff')
                    os.system('say "你存在疲劳驾驶的可能，请小心驾驶"')

        else:
            alarm = False
            # 如果连续2次都小于阈值，则表示进行了一次眨眼活动
            if COUNTER >= EYE_AR_CONSEC_FRAMES:  # 阈值：2
                TOTAL += 1
            # 重置眼帧计数器
            COUNTER = 0

        # 第十四步：进行画图操作，68个特征点标识
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

        # 第十五步：进行画图操作，同时使用cv2.putText将眨眼次数进行显示

        cv2.putText(frame, "Faces: {}".format(len(rects)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                    2)
        cv2.putText(frame, "Blinks: {}".format(TOTAL), (150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "COUNTER: {}".format(COUNTER), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                    2)
        cv2.putText(frame, "now: {}".format(now), (450, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "time: {}".format(time.strftime('%Y-%m-%d %H:%M:%S')), (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                    2)
    #print('眼睛实时长宽比:{:.2f} '.format(ear))
    if TOTAL > 20 :
        end = time.time()
        time_c = end - start
        if time_c <= 60:
            if not alarm:
                alarm = True
                cv2.putText(frame, "wake up!!", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(frame, "please push key s continue", (200, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 0, 255), 2)
                #for i in range(0, 1):
                    #os.system('say "你存在疲劳驾驶的可能，请停车休息"')
                if cv2.waitKey(1) & 0xFF == ord('s'):
                    TOTAL = 0
                    start = end
        else:
            TOTAL = 0
            start = end

    if TOTAL <= 10:
        end = time.time()
        time_c = end - start
        if time_c > 60:
             if not alarm:
                alarm = True
                cv2.putText(frame, "wake up!!", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 0, 255), 2)
                cv2.putText(frame, "please push key s continue", (200, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 0, 255), 2)
                #os.system('afplay /System/Library/Sounds/Sosumi.aiff')
                #for i in range(0,1):
                    #os.system('say "你存在疲劳驾驶的可能，请停车休息"')
                if cv2.waitKey(1) & 0xFF == ord('s'):
                    TOTAL = 0
                    start = end

       # cv2.putText(frame, "Blinks:{0}".format(COUNTER), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
       # cv2.putText(frame, "EAR:{:.2f}".format(EYE_AR_THRESH), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    # 窗口显示 show with opencv
    cv2.imshow("Frame", frame)

    # if the `q` key was pressed, break from the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头 release camera
cap.release()
# do a bit of cleanup
cv2.destroyAllWindows()
