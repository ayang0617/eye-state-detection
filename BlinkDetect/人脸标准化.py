#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
#@version: python3
#@author: duhanmin
#@contact: duhanmin@foxmail.com
#@software: PyCharm Community Edition
#@file: 人脸标准化.py
#@time: 2017/12/6 16:56
'''

#import face_recognition
from PIL import Image
import cv2
import dlib
import numpy as np
from IPython import embed
#通过三个点计算夹角
#b为夹角位置所在的点
#默认θ=1计算弧度，θ!不等于1时计算角度
def cos_angle(a,b,c):
    x,y = b-a,b-c
    Lx = np.sqrt(x.dot(x))   #a.dot(b) 与 np.dot(a,b)效果相同  , a b 均为矩阵
    Ly = np.sqrt(y.dot(y))
    cos_angle = x.dot(y)/(Lx*Ly)        #角的临边比斜边的值
    # 根据条件选择是计算弧度还是角度
    return np.arccos(cos_angle)*360/2/np.pi    #结果是反余弦函数

def normalization(path):

    detector = dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor(
        r'/Users/ayang/PycharmProjects/pythonProject/shape_predictor_68_face_landmarks.dat')
    img = cv2.imread(path)
    faces = detector(img, 1)
    feas = []  # 关键点

    top, bottom, left, right = 0, 0, 0, 0

    if (len(faces) > 0):
        for k, d in enumerate(faces):       #enenumerate表示枚举  显示格式是 编号,元素 比如:0 one

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
    d=feas[45, :][0]
    e=feas[36, :][1]

    a, b, c = feas[45, :], feas[36, :], np.array([d,e])
    angle=cos_angle(a,b,c)

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

    #先旋转在截图,因为先截图的话,没办法保证人脸是全的

    dst = img[top:bottom,left:right]

    matRotate = cv2.getRotationMatrix2D(((right-left)/2,(bottom-top)/2),angle,1)      #变换矩阵
    rot_img = cv2.warpAffine(dst,matRotate,((right-left),(bottom-top)))
    # cv2.imshow("rot_img",rot_img)
    # cv2.waitKey(0)

    res_dst = cv2.resize(rot_img,(50,50))

    cv2.imshow("res_dst",rot_img)
    cv2.waitKey(0)

if __name__ == '__main__':
    path='/home/lcl/Pictures/zjl.jpeg'
    normalization(path)




# def normalization(input,output):
#     path =input
#     out_path = output
#
#     # 读取图片并识别人脸
#     #img = face_recognition.load_image_file(path)
#     #face_locations = tuple(list(face_recognition.face_locations(img)[0]))
#
#     # 重新确定切割位置并切割
#     top = face_locations[0]
#     right = face_locations[1]
#     bottom = face_locations[2]
#     left = face_locations[3]
#     cutting_position = (left, top, right, bottom)
#     # 切割出人脸
#     im = Image.open(path)
#
#     region = im.crop(cutting_position)
#
#     # 人脸缩放
#     a = 50  # 人脸方格大小
#     if region.size[0] >= a or region.size[1] >= a:
#         region.thumbnail((a, a), Image.ANTIALIAS)
#     else:
#         region = region.resize((a, a), Image.ANTIALIAS)
#     # 人脸旋转
#     θ =trait_angle(path)
#     # region = region.rotate(θ)
#     # 保存人脸
#     region.save(out_path)
