import os
import time
import cv2
import numpy as np

def catch_frame(file_name, input_folder, window_name='Catch Your Face'):
    img_file = os.path.join(input_folder, file_name)
    if os.path.exists(img_file) == False:
        print("img_file not exist")
        return None
    # cv2.namedWindow(window_name)
    frame = cv2.imread(img_file)  # input which pic you want to detect
    # print(frame)
    if frame is None:
        return None
    else:
        if frame.any() == 0:
            return None
        else:
            # frame = cv2.resize(frame,(400,500))
            frame_size = frame.shape
            w = frame_size[1]
            h = frame_size[0]
            larger = 4000
            if(w>h):
              larger = w
            else:
              larger = h
            scale = 1
            if(larger>4000): 
                scale = 4000/larger
                # print("图像缩放")
                # print(scale)
                frame = cv2.resize(frame, None, fx=scale, fy=scale)
            frame = catch_face_frame(frame)
            return frame

def catch_face_frame(frame):
    # 告诉OpenCV使用人脸识别分类器
    classfier = cv2.CascadeClassifier('./cv2_detect/haarcascade_frontalface_alt2.xml')
    # 将当前帧转换成灰度图像
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 人脸检测，1.2和2分别为图片缩放比例和需要检测的有效点数
    # face_rects = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
    cv2.equalizeHist(grey, grey)
    face_rects = classfier.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=0, minSize=(128, 128))
    w1 = 0
    h1 = 0
    if len(face_rects) > 0:
          for face_rect in face_rects:
             x, y, w, h = face_rect
             # print(face_rect)
             size = w*h
             size1 = w1*h1
             if(size>size1):
                faceframe = frame[y:y + h, x:x + w]
                w1=w
                h1=h
          # resultframe = cv2pil(faceframe)
          # print(w1)
          # print(h1)
          return faceframe
    else:
         return None

def detect_loop():
    for i in range(2):
        if i == 0:
            input_folder = './pic/student/'
            output_folder = './data_face/student/Image/'
            # if not os.path.isdir(output_folder):
            #     os.makedirs(output_folder)
            all_imgs = os.listdir(input_folder)
            print(len(all_imgs))
            # for num in range(len(all_imgs)-1):
            num = 0
            while num < len(all_imgs):
                frame = catch_frame(all_imgs[num], input_folder)
                if frame is None:
                    num = num+1
                    continue
                if frame.any() == 0:
                    num = num + 1
                    continue
                save_path = '{}'.format(output_folder) + 'out_' + all_imgs[num]
                cv2.imwrite(save_path, frame)
                print("detect student images:{}".format(num))
                num = num+1
        if i == 1:
            input_folder = './pic/stranger/'
            output_folder = './data_face/stranger/Image/'
            # if not os.path.isdir(output_folder):
            #     os.makedirs(output_folder)
            all_imgs = os.listdir(input_folder)
            print(len(all_imgs))
            for num in range(len(all_imgs)):
                frame = catch_frame(all_imgs[num], input_folder)
                if frame is None:
                    continue
                if frame.any() == 0:
                    continue
                save_path = os.path.join(output_folder, 'output_' + all_imgs[num])
                cv2.imwrite(save_path, frame)
                print("detect stranger images:{}".format(num))

def detect_loop_predict():
    for i in range(2):
        if i == 0:
            input_folder = './predict_pic/student/'
            output_folder = './predict_face/student/Image/'
            # if not os.path.isdir(output_folder):
            #     os.makedirs(output_folder)
            all_imgs = os.listdir(input_folder)
            print(len(all_imgs))
            # for num in range(len(all_imgs)-1):
            num = 0
            while num < len(all_imgs):
                frame = catch_frame(all_imgs[num], input_folder)
                if frame is None:
                    num = num+1
                    continue
                if frame.any() == 0:
                    num = num + 1
                    continue
                save_path = '{}'.format(output_folder) + 'student_' + all_imgs[num]
                cv2.imwrite(save_path, frame)
                print("detect student images:{}".format(num))
                num = num+1
        if i == 1:
            input_folder = './predict_pic/stranger/'
            output_folder = './predict_face/stranger/Image/'
            # if not os.path.isdir(output_folder):
            #     os.makedirs(output_folder)
            all_imgs = os.listdir(input_folder)
            print(len(all_imgs))
            num = 0
            while num < len(all_imgs):
                frame = catch_frame(all_imgs[num], input_folder)
                if frame is None:
                    num = num + 1
                    continue
                if frame.any() == 0:
                    num = num + 1
                    continue
                save_path = '{}'.format(output_folder) + 'stranger_' + all_imgs[num]
                cv2.imwrite(save_path, frame)
                print("detect stranger images:{}".format(num))
                num = num + 1

def face_detect_and_cut(src_file, target_file):
    print("%s -> %s" %(src_file, target_file))

    frame = cv2.imread(src_file)

    if frame is None:
        return False

    else:
        if frame.any() == 0:
            return False

        else:
            frame = catch_face_frame(frame)

    if frame is None:
        return False

    if frame.any() == 0:
        return False

    return cv2.imwrite(target_file, frame)


if __name__ == '__main__':
    # detect_loop()
    detect_loop_predict()
