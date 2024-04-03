'''
功能：将视频逐帧抽取，在文件夹中保存为图片，可设置间隔帧数进行抽取，可设置图片名
'''
 
import os
import cv2

duration = 45  # 设置帧间隔

Video_Dir = "./20240403mao/mao_1080p_30f.mp4"    
Save_Dir = f'./20240403mao/mao_{duration}/images/'

def rotate_img(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h))
    return img

def video2images(Video_Dir):
 
    cap = cv2.VideoCapture(Video_Dir)
    c = 1  # 帧数起点
    index = 1  # 图片命名起点，如1.jpg
 
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
 
    while True:
        # 逐帧捕获
        ret, frame = cap.read()
        # 如果正确读取帧，ret为True
        if not ret:
            print("Can't receive frame.")
            break
        # 设置每5帧取一次图片，若想逐帧抽取图片，可设置c % 1 == 0
        if c % duration == 0:
            # 图片存放路径，即图片文件夹路径
            # cv2.imwrite(Save_Dir + 'main_' + str(index) + '.png', frame) 
            
            frame = rotate_img(frame, 180)
            
            cv2.imwrite(Save_Dir + 'main_' + '{:0>5d}'.format(index) + '.png', frame) 
            print('save image:', c, index)
            index += 1
            
        c += 1
        cv2.waitKey(1)
        # 按键停止
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
 
             # 视频存放路径
os.makedirs(Save_Dir, exist_ok=True)
video2images(Video_Dir)