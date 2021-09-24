import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import easyocr
import matplotlib.pyplot as plt  # plt 用于显示图片
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
VIDEO_DIR = "D:/dataset/video_3"
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


def get_frames(video_dir):
    get = 0
    miss = 0
    total_num = 0
    more = 0

    ocr = PaddleOCR(lang="en", gpu_mem=5000, det=False,
                    rec_model_dir="./inference/en_ppocr_mobile_v2.0_rec_infer/")  # 首次执行会自动下载模型文件

    # with open('video_6_score_change.json', 'r') as fd:
    #     Video_Score_dic = json.load(fd)
    SCORES = {}

    for root, _, files in os.walk(video_dir):
        for video_index in files:
            time1 = time.time()
            video_index1 = video_index.split('_')
            # score_change_dic = Video_Score_dic[video_index1[1]]
            # if not score_change_dic:
            #     continue
            # total_num += len(score_change_dic)
            score_dic = {}
            score_record = []
            videoCap = cv.VideoCapture(video_dir + "\\" + video_index)
            frame_height = videoCap.get(cv.CAP_PROP_FRAME_HEIGHT)
            frame_width = videoCap.get(cv.CAP_PROP_FRAME_WIDTH)
            h_index = int(frame_height * 7 / 10)
            w_index = int(frame_width / 2)
            frame_count = videoCap.get(cv.CAP_PROP_FRAME_COUNT)
            print("-----------------------------")
            print("video:{}".format(video_index1[1]))
            i = 0
            init_candidate = defaultdict(int)
            big_candidate = defaultdict(int)
            big_player_dic = {}
            init_flag = False
            k = 0
            location = []
            goal_index = 0
            mode = 0
            modify_candidate = defaultdict(int)
            modify_flag = False
            while (i < frame_count):
                # 评估候选比分
                for key in big_candidate.keys():
                    # 如果某个比分出现次数大于3
                    if big_candidate[key] >= 3:
                        big_player_dic[i] = key

                for key in init_candidate.keys():
                    # 如果某个比分出现次数大于3
                    if init_candidate[key] >= 3:
                        # 记录该比分与对应帧数
                        score_dic[i] = key
                        # 如果score_record里面暂时没有记录，就记录
                        if not score_record:
                            score_record.append((i, key))
                            # 如果没有初始位置，就记录
                            if not location:
                                # location = copy.deepcopy(temp_result[0][goal_index])
                                # location.append(temp_result[0][goal_index])
                                gt_x1 = temp_result[0][goal_index][0][0]
                                gt_x2 = temp_result[0][goal_index][1][0]
                                gt_y1 = temp_result[0][goal_index][0][1]
                                gt_y2 = temp_result[0][goal_index][2][1]
                        # 如果score_record已有记录，则判断新比分的合法性
                        else:
                            # 新比分
                            num1, num2 = key.split("-")
                            team1 = int(num1)
                            team2 = int(num2)
                            # 旧比分
                            a, b = score_record[-1]
                            num1, num2 = b.split("-")
                            gt_team1 = int(num1)
                            gt_team2 = int(num2)
                            cha1 = team1 - gt_team1
                            cha2 = team2 - gt_team2
                            # 如果直接合法，就记录到score_record
                            if (cha1 == 1 and cha2 == 0) or (cha1 == 0 and cha2 == 1):
                                score_record.append((i, key))
                                # 清空modify_candidate，准备检查刚进去的新比分
                                modify_candidate.clear()
                                print("get new score! i:{}  score:{}".format(i, key))
                                more += 1
                            # 如果不合法，但是有不同，则加入modify_candidate
                            elif cha1 != 0 or cha2 != 0:
                                modify_candidate[key] += 1
                                # 如果modify_candidate中某个比分出现超过4次，则说明score_record中记录的上一个比分有误，需要纠错
                                for key2 in modify_candidate.keys():
                                    if modify_candidate[key2] >= 4:
                                        a, b = score_record[-1]
                                        score_record[-1] = (a, key2)
                                        modify_flag = True
                                        break
                                if modify_flag:
                                    modify_candidate.clear()
                                    modify_flag = False
                            # 如果候选比分和上次记录的比分一致，则说明上次记录的是对的，清空modify_candidate，防止误判
                            elif cha1 == 0 and cha2 == 0:
                                modify_candidate.clear()

                        # 处理完出现超过3次的候选比分后，清空init_candidate，跳过500帧获取下一次比分
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
                    bigframe = temp_jpgframe[h_index:]
                    # 得到OCR识别结果

                    temp_result = ocr(jpgframe)
                    big_temp_result = ocr(bigframe)



                    big_result = ''
                    big_ocr_list = []
                    for mystr, ration in big_temp_result[1]:
                        big_result += mystr
                        big_ocr_list.append(mystr)
                        big_result += ' '

                    # print(big_result)
                    # print(big_ocr_list)

                    if big_result:
                        game_t = re.findall("\d\d:\d\d", big_result)
                        if game_t:
                            for ii, [strr, ration] in enumerate(big_temp_result[1]):
                                sc = re.findall("\d-\d", strr)
                                if sc:
                                    continue
                                game_tt = re.findall("\d\d:\d\d", strr)
                                if game_tt:
                                    continue
                                player_score = re.findall("\D+\d+\D*|\D*\d+\D+", strr)
                                if player_score and len(player_score[0]) > 4:
                                    big_candidate[player_score] += 1
                                    print("check players string: {}   frame:{}   time:{}".format(player_score, i,
                                                                                                 game_t))

                        # a = re.findall("\d+\'\D+\s", big_result)
                        # print(a)
                        # print(big_result)

                    result = ''
                    ocr_list = []
                    for mystr, ration in temp_result[1]:
                        result += mystr
                        ocr_list.append(mystr)
                        result += ' '

                    # 检索比赛时间
                    # mode:  0：初始状态，不确定比分啥形式    1：“-”形式   2：“:”形式
                    if result:
                        # “-”形式，直接找比分
                        if mode == 0 or mode == 1:
                            score2 = re.findall("\d-\d", result)
                            if score2:
                                num_num = score2[0]
                                init_candidate[num_num] += 1
                                i += 1
                                mode = 1
                                continue
                        # “:”形式，先排除时间，再找比分
                        if mode == 0 or mode == 2:
                            for ii, [strr, ration] in enumerate(temp_result[1]):
                                game_time = re.findall("\d\d:\d\d", strr)
                                # 找到时间给他归零
                                if game_time:
                                    temp_result[1][ii] = ("0", 0)
                            # 找比分，同时定位
                            for ii, [strr, ration] in enumerate(temp_result[1]):
                                game_time = re.findall("\d:\d", strr)
                                if game_time:
                                    num_num = game_time[0]
                                    # 如果已经记录过位置，则与记录位置进行比较
                                    if location:
                                        x1 = temp_result[0][ii][0][0]
                                        x2 = temp_result[0][ii][1][0]
                                        y1 = temp_result[0][ii][0][1]
                                        y2 = temp_result[0][ii][2][1]
                                        # 偏移量小于8则说明位置正确
                                        if (abs(gt_x1 - x1) + abs(gt_x2 - x2) + abs(gt_y1 - y1) + abs(gt_y2 - y2) < 8):
                                            num1, num2 = num_num.split(":")
                                            init_candidate[num1 + "-" + num2] += 1
                                            # 记录第一次出现这个比分的帧数
                                            goal_index = ii
                                            i += 1
                                            mode = 2
                                            goal_flag = True
                                            break
                                    # 没记录过，则只记录在init_candidate
                                    else:
                                        num1, num2 = num_num.split(":")
                                        init_candidate[num1 + "-" + num2] += 1
                                        goal_index = ii
                                        i += 1
                                        mode = 2
                                        goal_flag = True
                                        break

                    # 如果检测出比分，则帧数+1检测下一帧，否则帧数+200
                    if goal_flag:
                        continue
                    else:
                        i += 100
                        continue
                else:
                    print("can not get frames!")
                i += 100
            print(score_dic)
            print(score_record)
            time2 = time.time()
            print("time for this video:{}".format(time2 - time1))

            SCORES[video_index] = score_record

            with open('scores_3' + '.json', 'w') as fd:
                json.dump(SCORES, fd)

    print("total_num:{}   get:{}   miss:{}  more:{}".format(total_num, get, miss, more))


# FRAMES_DIC, NUM = get_frame_dic(base_dir=EVENT_DIR, classes='event', keyword=keyword1)
# print(FRAMES_DIC)
# print(NUM)
get_frames(video_dir=VIDEO_DIR)
