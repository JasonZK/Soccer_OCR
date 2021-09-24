import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import re
from paddleocr import PaddleOCR, draw_ocr
import cv2 as cv
import numpy as np
import time
from collections import defaultdict
import json
from fuzzywuzzy import fuzz
from fuzzywuzzy import process


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

keyword1 = "goal"


def makedir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)


EVENT_DIR = "D:/dataset/event"
VIDEO_DIR = "D:/dataset/video_error"

ocr = PaddleOCR(lang="en", gpu_mem=5000, det=False,
                rec_model_dir="./inference/en_ppocr_mobile_v2.0_rec_infer/")  # 首次执行会自动下载模型文件

# 获取队名文件
team_name = []
f = open("team_name.txt", "r")
for line in f:
    line = line[:-1]
    team_name.append(line)


# result = process.extract(a, team_name, scorer=fuzz.token_set_ratio, limit=5)

def evaluate_candidates(i, init_candidate, score_dic, score_record, location, temp_result, tempResult_index,
                        modify_candidate,
                        more, gt_y2,
                        big_candidate, player_score_dic, player_score_record, big_flag, player_name_dic):
    # for key in big_candidate.keys():
    #     # 如果某个比分出现次数大于3
    #     if big_candidate[key] >= 3:
    #         big_player_dic[i] = key


    # 评估候选比分
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
                    # gt_x1 = temp_result[0][goal_index][0][0]
                    # gt_x2 = temp_result[0][goal_index][1][0]
                    # gt_y1 = temp_result[0][goal_index][0][1]
                    gt_y2 = temp_result[0][tempResult_index][2][1]
                    location = True
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
                    # 清空modify_candidate，准备检查接下来进去的新比分
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
                            modify_candidate.clear()
                            break
                # 如果候选比分和上次记录的比分一致，则说明上次记录的是对的，清空modify_candidate，防止误判
                elif cha1 == 0 and cha2 == 0:
                    modify_candidate.clear()

            # 处理完出现超过3次的候选比分后，清空init_candidate，跳过500帧获取下一次比分
            if not big_flag:
                i += 500
            else:
                i += 2
            # print("i: {}   score:{}".format(i, key))
            init_candidate.clear()
            break
    return i, init_candidate, score_dic, score_record, modify_candidate, more, location, gt_y2, big_candidate, player_score_dic, player_score_record, player_name_dic


def get_ocr_result(videoCap, i, h_index):
    result = ''
    ocr_list = []
    temp_result = []

    videoCap.set(cv.CAP_PROP_POS_FRAMES, i)
    # i += 100;
    boolFrame, matFrame = videoCap.read()

    if boolFrame:
        temp_jpgframe = np.asarray(matFrame)
        # 截取上面0-90区域进行OCR检测
        if h_index < 200:
            jpgframe = temp_jpgframe[0:h_index]
        else:
            jpgframe = temp_jpgframe[h_index:]

        # 得到OCR识别结果
        temp_result = ocr(jpgframe)

        for mystr, ration in temp_result[1]:
            result += mystr
            ocr_list.append(mystr)
            result += ' '
    return result, temp_result


def find_goal_location(temp_result, location, init_candidate, gt_y2):
    goal_flag = False
    tempResult_index = 0
    for ii, [strr, ration] in enumerate(temp_result[1]):
        game_time = re.findall("\d:\d", strr)
        if game_time:
            num_num = game_time[0]
            # 如果已经记录过位置，则与记录位置进行比较
            if location:
                # x1 = temp_result[0][ii][0][0]
                # x2 = temp_result[0][ii][1][0]
                # y1 = temp_result[0][ii][0][1]
                y2 = temp_result[0][ii][2][1]
                # 偏移量小于8则说明位置正确
                # if (abs(gt_x1 - x1) + abs(gt_x2 - x2) + abs(gt_y1 - y1) + abs(gt_y2 - y2) < 8):
                if abs(gt_y2 - y2) < 4:
                    num1, num2 = num_num.split(":")
                    init_candidate[num1 + "-" + num2] += 1
                    # tempResult_index = ii
                    goal_flag = True
                    break
            # 没记录过，则只记录在init_candidate
            else:
                num1, num2 = num_num.split(":")
                init_candidate[num1 + "-" + num2] += 1
                # 记录该字符串在temp_result中的索引
                tempResult_index = ii
                goal_flag = True
                break
    return init_candidate, goal_flag, tempResult_index


def get_frames(video_dir):
    get = 0
    miss = 0
    total_num = 0
    more = 0
    # with open('video_6_score_change.json', 'r') as fd:
    #     Video_Score_dic = json.load(fd)
    SCORES = {}


    for root, _, files in os.walk(video_dir):
        for video_index in files:
            time1 = time.time()
            video_index1 = video_index.split('_')

            if video_index1[1] not in ['0059', '0060', '0061', '0062',
                                       '0063', '0064', '0065', '0066', '0067', '0068', '0069', '0070',
                                       '0071', '0072', '0073', '1040', '1041', '1045', '1046', '1047', '1048',
                                       '1049', '1050', '1051', '1052', '1054',
                                       '1055', '1056', '1057', '1058', '1059', '1171',
                                       '1212', '1216', '1218', '1221', '1223', '1224', '1225',
                                       '1226', '1228', '1230', '1231', '1233', '1236', '1237', '1238', '1239', '1242', '1243', '1244', '1245']:

                score_dic = {}
                score_record = []
                player_score_dic = {}
                player_score_record = []
                videoCap = cv.VideoCapture(video_dir + "\\" + video_index)

                frame_height = videoCap.get(cv.CAP_PROP_FRAME_HEIGHT)
                frame_width = videoCap.get(cv.CAP_PROP_FRAME_WIDTH)
                big_h_index = int(frame_height * 7 / 10)
                big_w_index = int(frame_width / 2)
                x1_big_center = big_w_index - 50
                x2_big_center = big_w_index + 50

                frame_count = videoCap.get(cv.CAP_PROP_FRAME_COUNT)
                print("-----------------------------")
                print("video:{}".format(video_index1[1]))
                i = 0
                init_candidate = defaultdict(int)
                player_name_dic = defaultdict(int)
                score_time_dic = defaultdict(int)
                big_candidate = defaultdict(set)
                location = False
                tempResult_index = 0
                mode = 0
                modify_candidate = defaultdict(int)
                temp_result = []
                gt_y2 = 0
                left_team = ''
                right_team = ''
                team_y1 = 10000
                team_y2 = 0
                big_flag = False
                while i < frame_count:
                    # 评估候选比分


                    i, init_candidate, score_dic, score_record, \
                    modify_candidate, more, location, gt_y2, \
                    big_candidate, player_score_dic, player_score_record, player_name_dic = \
                        evaluate_candidates(i, init_candidate, score_dic, score_record, location, temp_result,
                                            tempResult_index,
                                            modify_candidate, more, gt_y2,
                                            big_candidate, player_score_dic, player_score_record, big_flag, player_name_dic)

                    # 获取ocr第i帧的直接结果temp_result， 以及字符串连接后的result
                    result, temp_result = get_ocr_result(videoCap, i, 125)
                    big_result, big_temp_result = get_ocr_result(videoCap, i, big_h_index)

                    goal_flag = False
                    big_flag = False
                    if big_result:
                        # 有时间说明是比分大字幕，而不是其他下方字幕
                        game_t = re.findall("\d\d:\d\d", big_result)
                        if game_t:
                            for ii, [strr, ration] in enumerate(big_temp_result[1]):
                                x1_big = big_temp_result[0][ii][0][0]
                                x2_big = big_temp_result[0][ii][2][0]
                                y1_big = big_temp_result[0][ii][0][1]
                                y2_big = big_temp_result[0][ii][3][1]
                                if not left_team or not right_team:
                                    team1 = process.extractOne(strr, team_name, scorer=fuzz.token_set_ratio, score_cutoff=70)
                                    if team1:
                                        if x1_big > big_w_index:
                                            right_team = team1[0]
                                        if x2_big < big_w_index:
                                            left_team = team1[0]
                                        team_y1 = min(team_y1, y1_big)
                                        team_y2 = max(team_y2, y2_big)
                                    continue
                                elif abs(y1_big - team_y1) + abs(y2_big - team_y2) < 4:
                                    continue



                                # 首先排除时间和比分的数字影响
                                sc = re.findall("\d-\d", strr)
                                if sc:
                                    # x1_big_goal = big_temp_result[0][ii][0][0]
                                    # x2_big_goal = big_temp_result[0][ii][3][0]
                                    continue
                                game_tt = re.findall("\d\d:\d\d", strr)
                                if game_tt:
                                    continue

                                # 运动员进球信息格式1：
                                # 1个或多个非数字+1个或多个数字+0个或多个非数字 |
                                # 0个或多个非数字+1个或多个数字+1个或多个非数字
                                # 长度要求大于4
                                player_score_list = re.findall("\D+\d+\D*|\D*\d+\D+", strr)
                                if player_score_list and len(player_score_list[0]) > 4:
                                    player_score_temp = player_score_list[0]

                                    string_list = re.findall("[A-Z][a-z]+", player_score_temp)
                                    number_list = re.findall("\d+", player_score_temp)

                                    if not string_list:
                                        continue
                                    player_name = string_list[0]
                                    player_name_dic[player_name] += 1

                                    for number_one in number_list:
                                        if len(number_one) < 4:
                                            big_candidate[player_name].add(number_one)
                                            score_time_dic[number_one] += 1
                                            big_flag = True
                                            print("check players name: {}  goal_time: {}  frame:{}   time:{}".format(player_name, big_candidate[player_name], i,
                                                                                                        game_t))
                                    continue

                                # 运动员进球信息格式2：
                                # 运动员字符串在数字字符串前一个
                                #
                                goal_time = re.findall("^\d+$|^\d+\'|^\d+\(", strr)

                                if goal_time and team_y2 and (team_y1 - y1_big > 5 or y2_big - team_y2 > 5):
                                    player_name, rat = big_temp_result[1][ii - 1]
                                    player_name_dic[player_name] += 1
                                    isTeam = process.extractOne(player_name, team_name, scorer=fuzz.token_set_ratio,
                                                               score_cutoff=70)
                                    if isTeam:
                                        continue

                                    number_one = re.findall("\d+", goal_time[0])

                                    if len(number_one[0]) < 4:
                                        big_candidate[player_name].add(number_one[0])
                                        score_time_dic[number_one[0]] += 1
                                        big_flag = True
                                        print("check players name: {}  goal_time: {}  frame:{}   time:{}".format(
                                            player_name, big_candidate[player_name], i,
                                            game_t))



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
                                    # 排除时间后找比分，并且定位检测框
                                    init_candidate, goal_flag, tempResult_index = find_goal_location(temp_result,
                                                                                                     location,
                                                                                                     init_candidate,
                                                                                                     gt_y2)
                                    break

                    # 如果检测出比分，则帧数+1检测下一帧，否则帧数+200
                    if goal_flag:
                        i += 1
                        mode = 2
                        continue
                    elif big_flag:
                        i += 1
                        continue
                    else:
                        i += 100
                        # print("can not get frames!")
                        continue

                print(score_dic)
                print(score_record)

                # 把big_candidate里的值（set）取交集，如果不为空，就在player_name_dic里面看谁的key次数多
                # 次数少的key就在big_candidate中删除
                nn = len(big_candidate)
                names = list(big_candidate.keys())
                score_times = list(score_time_dic.keys())
                for i in range(nn):
                    if big_candidate[names[i]] != set('0'):
                        for score_time in list(big_candidate[names[i]]):
                            if score_time_dic[score_time] < 5:
                                big_candidate[names[i]].remove(score_time)
                        if big_candidate[names[i]] == set():
                            big_candidate[names[i]] = set('0')
                        for j in range(i+1, nn):
                            if big_candidate[names[j]] != set('0'):
                                set_temp = big_candidate[names[i]] & big_candidate[names[j]]
                                if set_temp:
                                    if player_name_dic[names[i]] >= player_name_dic[names[j]]:
                                        big_candidate[names[j]] = set('0')
                                    else:
                                        big_candidate[names[i]] = set('0')
                                elif fuzz.token_set_ratio(names[i], names[j]) >= 70:
                                    big_candidate[names[i]] = big_candidate[names[i]].union(big_candidate[names[j]])
                                    big_candidate[names[i]] = set('0')

                # big_candidate里的值（set）为‘0’的key也应该删除
                for key in names:
                    if '0' in big_candidate[key]:
                        del big_candidate[key]
                    elif key.isdigit():
                        del big_candidate[key]



                print(big_candidate)
                time2 = time.time()
                print("time for this video:{}".format(time2 - time1))

                # SCORES[video_index] = score_record

                # with open('scores_3' + '.json', 'w') as fd:
                #     json.dump(SCORES, fd)

    print("total_num:{}   get:{}   miss:{}  more:{}".format(total_num, get, miss, more))


get_frames(video_dir=VIDEO_DIR)
