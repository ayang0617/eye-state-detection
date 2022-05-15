# 参考https://github.com/lincolnhard/head-pose-estimation
import os

import cv2
import dlib
import imutils
import numpy as np
from imutils import face_utils

"""
思路：
    第一步：2D人脸关键点检测；第二步：3D人脸模型匹配；
    第三步：求解3D点和对应2D点的转换关系；第四步：根据旋转矩阵求解欧拉角。
"""

"""
只要知道世界坐标系内点的位置、像素坐标位置和相机参数就可以搞定旋转和平移矩阵(OpenCV自带函数solvePnp())
"""

# 世界坐标系(UVW)：填写3D参考点，该模型参考http://aifi.isr.uc.pt/Downloads/OpenGL/glAnthropometric3DModel.cpp
object_pts = np.float32([[6.825897, 6.760612, 4.402142],  # 33左眉左上角
                         [1.330353, 7.122144, 6.903745],  # 29左眉右角
                         [-1.330353, 7.122144, 6.903745],  # 34右眉左角
                         [-6.825897, 6.760612, 4.402142],  # 38右眉右上角
                         [5.311432, 5.485328, 3.987654],  # 13左眼左上角
                         [1.789930, 5.393625, 4.413414],  # 17左眼右上角
                         [-1.789930, 5.393625, 4.413414],  # 25右眼左上角
                         [-5.311432, 5.485328, 3.987654],  # 21右眼右上角
                         [2.005628, 1.409845, 6.165652],  # 55鼻子左上角
                         [-2.005628, 1.409845, 6.165652],  # 49鼻子右上角
                         [2.774015, -2.080775, 5.048531],  # 43嘴左上角
                         [-2.774015, -2.080775, 5.048531],  # 39嘴右上角
                         [0.000000, -3.116408, 6.097667],  # 45嘴中央下角
                         [0.000000, -7.415691, 4.070434]])  # 6下巴角

# 相机坐标系(XYZ)：添加相机内参
K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
     0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
     0.0, 0.0, 1.0]  # 等价于矩阵[fx, 0, cx; 0, fy, cy; 0, 0, 1]
# 图像中心坐标系(uv)：相机畸变参数[k1, k2, p1, p2, k3]
D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]

# 像素坐标系(xy)：填写凸轮的本征和畸变系数
cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)

# 重新投影3D点的世界坐标轴以验证结果姿势
reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                           [10.0, 10.0, -10.0],
                           [10.0, -10.0, -10.0],
                           [10.0, -10.0, 10.0],
                           [-10.0, 10.0, 10.0],
                           [-10.0, 10.0, -10.0],
                           [-10.0, -10.0, -10.0],
                           [-10.0, -10.0, 10.0]])
# 绘制正方体12轴
line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
              [4, 5], [5, 6], [6, 7], [7, 4],
              [0, 4], [1, 5], [2, 6], [3, 7]]


def get_head_pose(shape):
    # 填写2D参考点，注释遵循https://ibug.doc.ic.ac.uk/resources/300-W/
    """
      17左眉左上角/21左眉右角/22右眉左上角/26右眉右上角/36左眼左上角/39左眼右上角/42右眼左上角/
      45右眼右上角/31鼻子左上角/35鼻子右上角/48左上角/54嘴右上角/57嘴中央下角/8下巴角
    """
    # 像素坐标集合
    image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                            shape[39], shape[42], shape[45], shape[31], shape[35],
                            shape[48], shape[54], shape[57], shape[8]])
    """
    用solvepnp或sovlepnpRansac，输入3d点、2d点、相机内参、相机畸变，输出r、t之后
    用projectPoints，输入3d点、相机内参、相机畸变、r、t，输出重投影2d点
    计算原2d点和重投影2d点的距离作为重投影误差
    """
    # solvePnP计算姿势——求解旋转和平移矩阵：
    # rotation_vec表示旋转矩阵，translation_vec表示平移矩阵，cam_matrix与K矩阵对应，dist_coeffs与D矩阵对应。
    _, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs)
    # projectPoints重新投影误差
    reprojectdst, _ = cv2.projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix, dist_coeffs)

    reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))  # 以8行2列显示

    # 计算欧拉角calc euler angle
    # 参考https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#decomposeprojectionmatrix
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)  # 罗德里格斯公式（将旋转矩阵转换为旋转向量）
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))  # 水平拼接，vconcat垂直拼接
    # eulerAngles –可选的三元素矢量，包含三个以度为单位的欧拉旋转角度
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)  # 将投影矩阵分解为旋转矩阵和相机矩阵

    return reprojectdst, euler_angle

# 瞌睡点头
HAR_THRESH = 0.3
NOD_AR_CONSEC_FRAMES = 3
# 初始化帧计数器和点头总数
hCOUNTER = 0
hTOTAL = 0

print("[INFO] loading facial landmark predictor...")
# 第一步：使用dlib.get_frontal_face_detector() 获得脸部位置检测器
detector = dlib.get_frontal_face_detector()
# 第二步：使用dlib.shape_predictor获得脸部特征位置检测器
predictor = dlib.shape_predictor('/Users/ayang/PycharmProjects/pythonProject/shape_predictor_68_face_landmarks.dat')

# 第四步：打开cv2 本地摄像头
cap = cv2.VideoCapture(0)

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



        # 第十二步：进行画图操作，用矩形框标注人脸
        left = rect.left()
        top = rect.top()
        right = rect.right()
        bottom = rect.bottom()
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 1)

        """
        瞌睡点头
        """
        # 第十五步：获取头部姿态
        reprojectdst, euler_angle = get_head_pose(shape)

        har = euler_angle[0, 0]  # 取pitch旋转角度
        if har > HAR_THRESH:  # 点头阈值0.3
            hCOUNTER += 1
        else:
            # 如果连续2次都小于阈值，则表示瞌睡点头一次
            if hCOUNTER >= NOD_AR_CONSEC_FRAMES:  # 阈值：2
                hTOTAL += 1
            # 重置点头帧计数器
            hCOUNTER = 0

        # 绘制正方体12轴
        #for start, end in line_pairs:
            #cv2.line(frame, reprojectdst[start], reprojectdst[end], (0, 0, 255))
        # 显示角度结果
        cv2.putText(frame, "X: " + "{:7.2f}".format(euler_angle[0, 0]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (0, 255, 0), thickness=2)  # GREEN
        cv2.putText(frame, "Y: " + "{:7.2f}".format(euler_angle[1, 0]), (150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (255, 0, 0), thickness=2)  # BLUE
        cv2.putText(frame, "Z: " + "{:7.2f}".format(euler_angle[2, 0]), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (0, 0, 255), thickness=2)  # RED
        cv2.putText(frame, "Nod: {}".format(hTOTAL), (450, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # 第十六步：进行画图操作，68个特征点标识
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)



    # 确定疲劳提示:眨眼50次，打哈欠15次，瞌睡点头15次
    if  hTOTAL >= 15:
        cv2.putText(frame, "SLEEP!!!", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
        for i in range(0, 2):
            os.system('say "你存在疲劳驾驶的可能，请停车休息"')
            hTOTAL = 0

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

