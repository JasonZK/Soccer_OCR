import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import easyocr
import matplotlib.pyplot as plt # plt 用于显示图片
import re
from paddleocr import PaddleOCR, draw_ocr
import cv2 as cv
import numpy as np
import time
from collections import defaultdict
import json
import copy

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

keyword1 = "goal"


# print(BASE_DIR)


def makedir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)


EVENT_DIR = "D:/dataset/event"
VIDEO_DIR = "D:/dataset/video_6"
# FRAMES_DIR = os.path.join(BASE_DIR, "goal_frames")


# 遍历event文件夹
VID1 = ['0003', '0029', '0056', '0072', '1003', '1010', '1133', '1158', '1208',
        '1230']




def get_frame_dic(base_dir, classes, keyword):
    frame_dic = {}
    num = 0
    for root, _, files in os.walk(base_dir):
        # 遍历每个txt
        for event_index in files:
            # 只处理event.txt
            if classes in event_index:
                # event_index[:4]获取视频序号
                with open(base_dir + "\\" + event_index, "r") as f:
                    for line in f.readlines():
                        line1 = line.strip('\n')  # 去掉列表中每一个元素的换行符
                        line = line.split()
                        if keyword in line1 and line[3] == '0':
                            frame_dic.setdefault(event_index[:4], []).append(line[1] + " " + line[2])
                            num += 1

    return frame_dic, num


def get_frames(video_dir, frame_dic, n):
    get = 0
    miss = 0
    total_num = 0
    more = 0

    ocr = PaddleOCR(lang="en", gpu_mem=5000, det=False,
                    rec_model_dir="./inference/en_ppocr_mobile_v2.0_rec_infer/")  # 首次执行会自动下载模型文件

    with open('video_6_score_change.json', 'r') as fd:
        Video_Score_dic = json.load(fd)

    for root, _, files in os.walk(video_dir):
        for video_index in files:
            time1 = time.time()
            video_index1 = video_index.split('_')
            score_change_dic = Video_Score_dic[video_index1[1]]
            if not score_change_dic:
                continue
            total_num += len(score_change_dic)
            score_dic = {}
            score_record = []
            videoCap = cv.VideoCapture(video_dir + "\\" + video_index)
            frame_count = videoCap.get(cv.CAP_PROP_FRAME_COUNT)
            print("-----------------------------")
            print("video:{}".format(video_index1[1]))
            i = 0
            init_candidate = defaultdict(int)
            init_flag = False
            k = 0
            location = []
            goal_index = 0
            mode = 0
            modify_candidate = defaultdict(int)
            modify_flag = False
            while (i < frame_count):
                for key in init_candidate.keys():
                    if init_candidate[key] >= 3:
                        num1, num2 = key.split("-")
                        team1 = int(num1)
                        team2 = int(num2)
                        score_dic[i] = key
                        if not score_record:
                            score_record.append((i, key))
                            if not location:
                                # location = copy.deepcopy(temp_result[0][goal_index])
                                # location.append(temp_result[0][goal_index])
                                gt_x1 = temp_result[0][goal_index][0][0]
                                gt_x2 = temp_result[0][goal_index][1][0]
                                gt_y1 = temp_result[0][goal_index][0][1]
                                gt_y2 = temp_result[0][goal_index][2][1]
                        else:
                            num1, num2 = key.split("-")
                            team1 = int(num1)
                            team2 = int(num2)
                            a, b = score_record[-1]
                            num1, num2 = b.split("-")
                            gt_team1 = int(num1)
                            gt_team2 = int(num2)
                            cha1 = team1 - gt_team1
                            cha2 = team2 - gt_team2
                            if (cha1 == 1 and cha2 == 0) or (cha1 == 0 and cha2 == 1):
                                score_record.append((i, key))
                                modify_candidate.clear()
                                if k < len(score_change_dic):
                                    gt = int(score_change_dic[k])
                                    if i >= gt-2000 and i <= gt+1500:
                                        print("check!  gt:{}  i:{}  score:{}".format(gt, i, key))
                                        get += 1
                                        k += 1
                                    else:
                                        print("miss! gt:{}  i:{}  score:{}".format(gt, i, key))
                                        miss += 1
                                        k += 1
                                else:
                                    print("seem to be more ! gt:{}  i:{}  score:{}".format(gt, i, key))
                                    more += 1
                            elif cha1 != 0 or cha2 != 0:
                                modify_candidate[key] += 1
                                for key2 in modify_candidate.keys():
                                    if modify_candidate[key2] >= 4:
                                        a, b = score_record[-1]
                                        score_record[-1] = (a, key2)
                                        modify_flag = True
                                        break
                                if modify_flag:
                                    modify_candidate.clear()
                                    modify_flag = False
                            elif cha1 == 0 and cha2 == 0:
                                modify_candidate.clear()


                        i += 500
                        # print("i: {}   score:{}".format(i, key))
                        init_candidate.clear()
                        init_flag = True
                        break
                videoCap.set(cv.CAP_PROP_POS_FRAMES, i)
                # i += 100;
                boolFrame, matFrame = videoCap.read()
                goal_flag = False
                if boolFrame:
                    temp_jpgframe = np.asarray(matFrame)
                    # 截取上面0-90区域进行OCR检测
                    jpgframe = temp_jpgframe[0:125]
                    # 得到OCR识别结果

                    temp_result = ocr(jpgframe)

                    result = ''
                    ocr_list = []
                    for mystr, ration in temp_result[1]:
                        result += mystr
                        ocr_list.append(mystr)
                        result += ' '

                    # 检索比赛时间
                    if result:
                        if mode == 0 or mode == 1:
                            score2 = re.findall("\d-\d", result)
                            if score2:
                                num_num = score2[0]
                                init_candidate[num_num] += 1
                                i += 1
                                mode = 1
                                continue
                        if mode == 0 or mode == 2:
                            for ii, [strr, ration] in enumerate(temp_result[1]):
                                game_time = re.findall("\d\d:\d\d", strr)
                                if game_time:
                                    temp_result[1][ii] = ("0", 0)


                            for ii, [strr, ration] in enumerate(temp_result[1]):
                                game_time = re.findall("\d:\d", strr)
                                if game_time:
                                    num_num = game_time[0]
                                    if location:
                                        x1 = temp_result[0][ii][0][0]
                                        x2 = temp_result[0][ii][1][0]
                                        y1 = temp_result[0][ii][0][1]
                                        y2 = temp_result[0][ii][2][1]
                                        if (abs(gt_x1 - x1) + abs(gt_x2 - x2) + abs(gt_y1 - y1) + abs(gt_y2 - y2) < 8):
                                            num1, num2 = num_num.split(":")
                                            init_candidate[num1 + "-" + num2] += 1
                                            goal_index = ii
                                            i += 1
                                            mode = 2
                                            goal_flag = True
                                            break
                                    else:
                                        num1, num2 = num_num.split(":")
                                        init_candidate[num1 + "-" + num2] += 1
                                        goal_index = ii
                                        i += 1
                                        mode = 2
                                        goal_flag = True
                                        break
                    if goal_flag:
                        continue
                    else:
                        i += 200
                        continue
                else:
                    print("can not get frames!")
                i += 200
            print(score_dic)
            print(score_record)
            time2 = time.time()
            print("time for this video:{}".format(time2 - time1))
    print("total_num:{}   get:{}   miss:{}  more:{}".format(total_num, get, miss, more))





FRAMES_DIC, NUM = get_frame_dic(base_dir=EVENT_DIR, classes='event', keyword=keyword1)
# print(FRAMES_DIC)
# print(NUM)
get_frames(video_dir=VIDEO_DIR, frame_dic=FRAMES_DIC, n=NUM)
