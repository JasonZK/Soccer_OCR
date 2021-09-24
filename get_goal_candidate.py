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
    num = 0
    Video_Change_Dic = {}

    ocr = PaddleOCR(lang="en", gpu_mem=5000, det=False,
                    rec_model_dir="./inference/en_ppocr_mobile_v2.0_rec_infer/")  # 首次执行会自动下载模型文件

    for root, _, files in os.walk(video_dir):
        for video_index in files:
            video_index1 = video_index.split('_')
            # print(video_index1)
            change_dic = []
            if video_index1[1] in frame_dic.keys():
                videoCap = cv.VideoCapture(video_dir + "\\" + video_index)
                print("-----------------------------")
                print("video:{}".format(video_index1[1]))
                for index in frame_dic[video_index1[1]]:
                    num += 1
                    start = int(index.split()[0])
                    end = int(index.split()[1])
                    print("start:{}   end:{}".format(start, end))
                    i = start
                    flag = False
                    team1 = 0
                    team2 = 0
                    miss_str = []
                    init_candidate = defaultdict(int)
                    init_flag = False
                    while i <= start + 50:
                        for key in init_candidate.keys():
                            if init_candidate[key] >= 3:
                                num1, num2 = key.split("-")
                                team1 = int(num1)
                                team2 = int(num2)
                                i += 1
                                init_flag = True
                                break
                        if init_flag:
                            break
                        videoCap.set(cv.CAP_PROP_POS_FRAMES, i)
                        # print("I'm finding   video:{} frame:{}".format(video_index1[1], i))
                        boolFrame, matFrame = videoCap.read()
                        if boolFrame:
                            temp_jpgframe = np.asarray(matFrame)
                            # 截取上面0-90区域进行OCR检测
                            jpgframe = temp_jpgframe[0:125]
                            # 得到OCR识别结果
                            time1 = time.time()
                            temp_result = ocr(jpgframe)
                            time2 = time.time()
                            # print("    time for one:{}".format(time2 - time1))
                            result = ''
                            for mystr, ration in temp_result[1]:
                                result += mystr
                                result += ' '
                            # result = ''.join(re_str)
                            # print(result)
                            # 检索比赛时间
                            if result:
                                score2 = re.findall("\d-\d", result)
                                if score2:
                                    num_num = score2[0]
                                    # num1, num2 = num_num.split("-")
                                    # team1 = int(num1)
                                    # team2 = int(num2)
                                    # i += 1
                                    init_candidate[num_num] += 1
                                    i += 1
                                    continue
                                game_time = re.findall("\d\d:\d\d", result)
                                # print(game_time)
                                # 获取删除比赛时间之后的字符串
                                if not game_time:
                                    i += 1
                                    continue
                                temp = result.replace(game_time[0], '')
                                if temp:
                                    # 如果不为空，则检索比分
                                    score1 = re.findall("\d:\d", temp)
                                    if score1:
                                        num_num = score1[0]
                                        num1, num2 = num_num.split(":")
                                        init_candidate[num1 + "-" + num2] += 1
                                        i += 1
                                        continue
                                        # num1, num2 = num_num.split(":")
                                        # team1 = int(num1)
                                        # team2 = int(num2)
                                        # i += 1
                                    else:
                                        i += 1
                                        continue
                        else:
                            print("can not get frames!")
                        i += 1
                    print("  find init: {}-{}  i:{}".format(team1, team2, i))
                    score_candidate = defaultdict(int)
                    while i <= end + 4000:
                        for key in score_candidate.keys():
                            if score_candidate[key] >= 3:
                                num1, num2 = key.split("-")
                                num1 = int(num1)
                                num2 = int(num2)
                                print("  score change! before: {}-{}  after: {}-{}  i:{}".format(team1, team2, num1, num2, i))
                                change_dic.append(i)
                                get += 1
                                flag = True
                                break
                        if flag:
                            break
                        videoCap.set(cv.CAP_PROP_POS_FRAMES, i)
                        # print("I'm working with   video:{} frame:{}".format(video_index1[1], i))
                        boolFrame, matFrame = videoCap.read()
                        if boolFrame:
                            temp_jpgframe = np.asarray(matFrame)
                            # 截取上面0-90区域进行OCR检测
                            jpgframe = temp_jpgframe[0:125]
                            # 得到OCR识别结果
                            time1 = time.time()
                            temp_result = ocr(jpgframe)
                            time2 = time.time()
                            # print("    time for one:{}".format(time2 - time1))
                            result = ''
                            for mystr, ration in temp_result[1]:
                                result += mystr
                                result += ' '
                            # result = ''.join(re_str)
                            # print(result)
                            # 检索比赛时间
                            miss_str.append(result)
                            if result:
                                score2 = re.findall("\d-\d", result)
                                if score2:
                                    num_num = score2[0]
                                    num1, num2 = num_num.split("-")
                                    num1 = int(num1)
                                    num2 = int(num2)
                                    i += 100
                                    if team1 != num1 or team2 != num2:
                                        score_candidate[num_num] += 1
                                        i += 10
                                        # print("  score change! before: {}-{}  after: {}-{}   str:{}".format(team1, team2, num1, num2, result))
                                        # get += 1
                                        # flag = True
                                        # break
                                        continue
                                    else:
                                        i += 100
                                        continue
                                game_time = re.findall("\d\d:\d\d", result)
                                # print(game_time)
                                # 获取删除比赛时间之后的字符串
                                if not game_time:
                                    i += 100
                                    continue
                                temp = result.replace(game_time[0], '')
                                if temp:
                                    # 如果不为空，则检索比分
                                    score1 = re.findall("\d:\d", temp)
                                    if score1:
                                        num_num = score1[0]
                                        num1, num2 = num_num.split(":")
                                        num1 = int(num1)
                                        num2 = int(num2)
                                    else:
                                        i += 100
                                        continue
                                    if team1 != num1 or team2 != num2:
                                        score_candidate[str(num1) + "-" + str(num2)] += 1
                                        i += 10
                                        # print("  score change! before: {}-{}  after: {}-{}   str:{}".format(team1, team2, num1, num2, result))
                                        # get += 1
                                        # flag = True
                                        continue
                                    else:
                                        i += 100
                                        continue
                        else:
                            print("can not get frames!")
                        i += 100
                    if not flag:
                        miss += 1
                        print("  miss!  goal frame:{}  {}".format(start, end))
                        for str_mis in miss_str:
                            print("    {}".format(str_mis))
            Video_Change_Dic[video_index1[1]] = change_dic

    with open('video_6_score_change' + '.json', 'w') as fd:
        json.dump(Video_Change_Dic, fd)
    print("--------------------------------------------")
    print("total goal:{}   get:{}   miss:{}".format(num, get, miss))



FRAMES_DIC, NUM = get_frame_dic(base_dir=EVENT_DIR, classes='event', keyword=keyword1)
print(FRAMES_DIC)
print(NUM)
get_frames(video_dir=VIDEO_DIR, frame_dic=FRAMES_DIC, n=NUM)
