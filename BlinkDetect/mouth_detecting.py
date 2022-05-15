# -*- coding: utf-8 -*-
# import the necessary packages
import os

from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np  # 数据处理的库 numpy
import argparse
import imutils
import time
import dlib
import cv2

def mouth_aspect_ratio(mouth):
    A = np.linalg.norm(mouth[2] - mouth[9])  # 51, 59
    B = np.linalg.norm(mouth[4] - mouth[7])  # 53, 57
    C = np.linalg.norm(mouth[0] - mouth[6])  # 49, 55
    mar = (A + B) / (2.0 * C)
    return mar



# 打哈欠长宽比
# 闪烁阈值
MAR_THRESH = 0.80
MOUTH_AR_CONSEC_FRAMES = 4
# 初始化帧计数器和打哈欠总数
mCOUNTER = 0
mTOTAL = 0

# 初始化DLIB的人脸检测器（HOG），然后创建面部标志物预测
print("[INFO] loading facial landmark predictor...")
# 第一步：使用dlib.get_frontal_face_detector() 获得脸部位置检测器
detector = dlib.get_frontal_face_detector()
# 第二步：使用dlib.shape_predictor获得脸部特征位置检测器
predictor = dlib.shape_predictor('/Users/ayang/PycharmProjects/pythonProject/shape_predictor_68_face_landmarks.dat')

# 第三步：分别获取左右眼面部标志的索引
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# 第四步：打开cv2 本地摄像头
cap = cv2.VideoCapture(0)

# 从视频流循环帧
while True:
    # 第五步：进行循环，读取图片，并对图片做维度扩大，并进灰度化
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=720)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 第六步：使用detector(gray, 0) 进行脸部位置检测
    rects = detector(gray, 0)

    # 第七步：循环脸部位置信息，使用predictor(gray, rect)获得脸部特征位置的信息
    for rect in rects:
        shape = predictor(gray, rect)

        # 第八步：将脸部特征信息转换为数组array的格式
        shape = face_utils.shape_to_np(shape)

        # 第九步： 嘴巴坐标
        mouth = shape[mStart:mEnd]

        # 第十步：打哈欠
        mar = mouth_aspect_ratio(mouth)

        # 第十一步：使用cv2.convexHull获得凸包位置，使用drawContours画出轮廓位置进行画图操作
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

        # 第十二步：进行画图操作，用矩形框标注人脸
        left = rect.left()
        top = rect.top()
        right = rect.right()
        bottom = rect.bottom()
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)

        '''
            计算张嘴评分，如果小于阈值，则加1，如果连续3次都小于阈值，则表示打了一次哈欠，同一次哈欠大约在3帧
        '''
        # 同理，判断是否打哈欠
        if mar > MAR_THRESH:  # 张嘴阈值0.10
            mCOUNTER += 1
            cv2.putText(frame, "Yawning!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # 如果连续6次都小于阈值，则表示打了一次哈欠
            if mCOUNTER >= MOUTH_AR_CONSEC_FRAMES:  # 阈值：6
                mTOTAL += 1
            # 重置嘴帧计数器
            mCOUNTER = 0
        cv2.putText(frame, "Yawning: {}".format(mTOTAL), (150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "mCOUNTER: {}".format(mCOUNTER), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "MAR: {:.2f}".format(mar), (480, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 第十五步：进行画图操作，68个特征点标识
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

#    print('嘴巴实时长宽比:{:.2f} '.format(mar) + "\t是否张嘴：" + str([False, True][mar > MAR_THRESH]))

    # 确定疲劳提示
    if mTOTAL >= 15:
        cv2.putText(frame, "STOP!!!", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        for i in range(0, 2):
            os.system('say "你存在疲劳驾驶的可能，请停车休息"')
            mTOTAL = 0

    # 按q退出
    cv2.putText(frame, "Press 'q': Quit", (20, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (84, 255, 159), 2)
    # 窗口显示 show with opencv
    cv2.imshow("Frame", frame)

    # if the `q` key was pressed, break from the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头 release camera
cap.release()
# do a bit of cleanup
cv2.destroyAllWindows()
