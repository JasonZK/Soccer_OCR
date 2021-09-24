import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from paddleocr import PaddleOCR, draw_ocr
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time

video_path = "D:/dataset/video_2/soccer_0020_video.mp4"
img_path = 'big_yingchao.png'
img_path2 = 'big_19yazhou.png'

videoCap = cv.VideoCapture(video_path)
videoCap.set(cv.CAP_PROP_POS_FRAMES, 61307)

frame_height = videoCap.get(cv.CAP_PROP_FRAME_HEIGHT)
frame_width = videoCap.get(cv.CAP_PROP_FRAME_WIDTH)
h_index = int(frame_height * 7 / 10)
w_index = int(frame_width / 2)

boolFrame, matFrame = videoCap.read()
image = cv.imread(img_path)
# image = image[0:120]
image2 = cv.imread(img_path2)
# image2 = image2[0:120]
if boolFrame:
    temp_jpgframe = np.asarray(matFrame)
    # # 截取上面0-90区域进行OCR检测
    # jpgframe = temp_jpgframe[h_index:, 0:w_index]
    jpgframe = temp_jpgframe[h_index:]

    # jpgframe = temp_jpgframe
    # matFrame = matFrame[0:120]
    plt.imshow(jpgframe)
    plt.show()

    # 同样也是通过修改 lang 参数切换语种
    # ocr = PaddleOCR(det=False, gpu_mem=5000,
    #                 det_model_dir="./inference/ch_ppocr_server_v2.0_det_infer/",
    #                 rec_model_dir="./inference/ch_ppocr_server_v2.0_rec_infer/") # 首次执行会自动下载模型文件

    ocr = PaddleOCR(lang="en", gpu_mem=5000, det=False,
                    rec_model_dir="./inference/en_ppocr_mobile_v2.0_rec_infer/")  # 首次执行会自动下载模型文件

    # 可通过参数控制单独执行识别、检测
    time1 = time.time()
    result = ocr(jpgframe)
    time2 = time.time()
    print(time2 - time1)
    # result = ocr.ocr(img_path, rec=False) 只执行检测
    # 打印检测框和识别结果
    re = []
    for str, ration in result[1]:
        re.append(str)
    # re = ' '.join(re)
    # print(result)
    print(re)


    # time1 = time.time()
    # result = ocr(image2)
    # time2 = time.time()
    # print(time2 - time1)
    # # result = ocr.ocr(img_path, rec=False) 只执行检测
    # # 打印检测框和识别结果
    # re = []
    # for str, ration in result[1]:
    #     re.append(str)
    #
    # print(re)
    # for line in result:
    #     print(line)
else:
    print("can not get frames!")