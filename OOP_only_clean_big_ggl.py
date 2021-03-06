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


class PlayerTime:
    def __init__(self, playername=''):
        self.goaltime = defaultdict(str)
        self.playername = playername
        self.location = 0
        self.use = 1


class OneLine:
    def __init__(self, y1, y2):
        self.y1 = y1
        self.y2 = y2
        self.strr = ""
        self.times = []
        self.location = ""


def makedir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)


EVENT_DIR = "D:/dataset/event"
VIDEO_DIR = "D:/dataset/video_big_2"

ocr = PaddleOCR(lang="en", gpu_mem=5000, det=False,
                rec_model_dir="./inference/en_ppocr_mobile_v2.0_rec_infer/")  # 首次执行会自动下载模型文件

# 获取队名文件
team_name = []
f = open("team_name.txt", "r")
for line in f:
    line = line[:-1]
    team_name.append(line)


# result = process.extract(a, team_name, scorer=fuzz.token_set_ratio, limit=5)

def print_json(data):
    print(json.dumps(data, sort_keys=True, indent=4, separators=(', ', ': '), ensure_ascii=False))

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


def check_line(result_line, LorR, player_name_dic, big_candidate, score_time_dic, i, Player_Time_list, big_flag):
    for lline in result_line:
        P = 0
        OG = 0
        score_time = '0'
        string_list = re.findall("[A-Za-z]+", lline.strr)
        number_list = re.findall("\d+", lline.strr)

        if (not string_list) or (not number_list):
            continue

        player_name = ''
        # 这里目前只处理了普通的P、90加时的P以及普通的OG
        # 还可能有45加时的P，以及45/90加时的OG
        for string_one in string_list:
            if string_one.upper() == 'P':
                P = 1
                P_add90_score_time_list = re.findall("90\D+\d\'? ?\(?P|90\D+\d\'? ?\[?P", lline.strr)
                if P_add90_score_time_list:
                    add_number = re.findall("\d", P_add90_score_time_list[0])[0]
                    score_time = '90+' + add_number
                score_time_list = re.findall("\d+\'? ?\(?P|\d+\'? ?\[?P", lline.strr)
                if score_time_list:
                    score_time = re.findall("\d+", score_time_list[0])[0]
            elif string_one.upper() == 'OG':
                OG = 1
                score_time_list = re.findall("\d+\'? ?\(?OG|\d+\'? ?\[?OG", lline.strr)
                if score_time_list:
                    score_time = re.findall("\d+", score_time_list[0])[0]
            else:
                player_name += (string_one + ' ')

        if not player_name or len(player_name) < 3:
            continue

        player_name_dic[player_name] += 1

        one_playertime = PlayerTime(player_name)
        one_playertime.location = LorR
        if P and score_time != '0':
            one_playertime.goaltime[score_time] = 'P'
        elif OG and score_time != '0':
            one_playertime.goaltime[score_time] = 'OG'
            one_playertime.location = -LorR

        add_number = ''
        for number_one in number_list:
            # 如果这个数字是90后面加的，就跳过
            if number_one == add_number:
                continue
            # 如果数字是90，说明有加时
            if number_one == '90' or number_one == '45':
                add_time = 1
                # add_score_time_list = re.findall("\+\d\'?", lline.strr)
                add90_score_time_list = re.findall("90\D+\d", lline.strr)
                add45_score_time_list = re.findall("45\D+\d", lline.strr)
                if add90_score_time_list:
                    add_number = add90_score_time_list[0][-1]
                    score_time = '90+' + add_number
                    # 如果这个进球不是P或者OG
                    if not one_playertime.goaltime[score_time]:
                        one_playertime.goaltime[score_time] = "add"
                elif add45_score_time_list:
                    add_number = add45_score_time_list[0][-1]
                    score_time = '45+' + add_number
                    # 如果这个进球不是P或者OG
                    if not one_playertime.goaltime[score_time]:
                        one_playertime.goaltime[score_time] = "add"
                else:
                    score_time = number_one
                    # 如果这个进球不是P或者OG
                    if not one_playertime.goaltime[score_time]:
                        one_playertime.goaltime[score_time] = "N"
                big_candidate[player_name].add(score_time)
                score_time_dic[score_time] += 1
                big_flag = True
            # 正常进球情况
            # 理论上应该小于3，但是有1302特殊情况时间有115，101
            elif len(number_one) < 4:
                score_time = number_one
                big_candidate[player_name].add(score_time)
                # 如果这个进球不是P或者OG
                if not one_playertime.goaltime[score_time]:
                    one_playertime.goaltime[score_time] = "N"
                score_time_dic[score_time] += 1
                big_flag = True

        Player_Time_list.append(one_playertime)
        print("source string: {}   check players name: {}  goal_time: {}  frame:{}  goaltime:{}".format(
            lline.strr, player_name, big_candidate[player_name], i, one_playertime.goaltime))
        del one_playertime
        continue

    return player_name_dic, big_candidate, score_time_dic, Player_Time_list, big_flag


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
                                       '1226', '1228', '1230', '1231', '1233', '1236', '1237', '1238', '1239', '1242',
                                       '1243', '1244', '1245']:

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

                i = frame_count - 25000
                # i = 0

                init_candidate = defaultdict(int)
                player_name_dic = defaultdict(int)
                score_time_dic = defaultdict(int)
                big_candidate = defaultdict(set)

                location_player_score = 0  # 得分球员位置，为0代表在队名下面，为1代表在队名上面
                tempResult_index = 0
                mode = 0
                modify_candidate = defaultdict(int)
                temp_result = []
                gt_y2 = 0
                left_team = ''
                right_team = ''
                left_team_x = 0
                right_team_x = 0
                team_y1 = 10000
                team_y2 = 0
                big_flag = False
                big_nums = 6
                Player_Time_list = []
                while i < frame_count:
                    # if i > 78800:
                    #     print("a")
                    # 评估候选比分

                    # 获取ocr第i帧的直接结果temp_result， 以及字符串连接后的result
                    result, temp_result = get_ocr_result(videoCap, i, 125)
                    big_result, big_temp_result = get_ocr_result(videoCap, i, big_h_index)

                    goal_flag = False
                    big_flag = False
                    team_nums = 0

                    left_line = []
                    right_line = []
                    left_big_temp_result = []
                    right_big_temp_result = []

                    if big_result:
                        # 首先检测是否存在两个队名
                        team_candidates_list = []
                        two_teams = False
                        for ii, [strr, ration] in enumerate(big_temp_result[1]):
                            x1_big = big_temp_result[0][ii][0][0]
                            x2_big = big_temp_result[0][ii][2][0]
                            y1_big = big_temp_result[0][ii][0][1]
                            y2_big = big_temp_result[0][ii][3][1]
                            # 排除在中间位置的字符串
                            if abs((x1_big + x2_big) / 2 - big_w_index) < 4:
                                continue

                            team1 = process.extractOne(strr, team_name, scorer=fuzz.ratio,
                                                       score_cutoff=70)
                            if team1:
                                if team_candidates_list:
                                    for team_candidate in team_candidates_list:
                                        # 两个检测出的队名必须满足：y坐标近似，x坐标关于中间对称
                                        if abs(team_candidate[2] - (y1_big + y2_big) / 2) < 3 and \
                                                abs((team_candidate[1] + (x1_big + x2_big) / 2) / 2 - big_w_index) < 10:
                                            two_teams = True
                                            if (x1_big + x2_big) / 2 > big_w_index:
                                                right_team = team1[0]
                                                left_team = team_candidate[0]
                                                # 确定队名的x坐标（中心）用于后面排除太过边缘的字符串
                                                left_team_x = team_candidate[1]
                                                right_team_x = (x1_big + x2_big) / 2
                                            if x2_big < big_w_index:
                                                left_team = team1[0]
                                                right_team = team_candidate[0]
                                                left_team_x = (x1_big + x2_big) / 2
                                                right_team_x = team_candidate[1]
                                            team_y1 = y1_big
                                            team_y2 = y2_big
                                            break
                                team_candidates_list.append([team1[0], (x1_big + x2_big) / 2, (y1_big + y2_big) / 2])
                                continue
                            if (x1_big + x2_big) / 2 < big_w_index:
                                left_big_temp_result.append([strr, x1_big, x2_big, y1_big, y2_big])
                            else:
                                right_big_temp_result.append([strr, x1_big, x2_big, y1_big, y2_big])

                        # 检测出了两个球队名，说明是大字幕，此时team_y1，team_y2都有了
                        if two_teams:
                            if left_big_temp_result:
                                sorted_left_big_result = sorted(left_big_temp_result,
                                                                key=lambda student: (student[3], student[2]))
                                yy1 = sorted_left_big_result[0][3]
                                yy2 = sorted_left_big_result[0][4]
                                left_oneline_0 = OneLine(sorted_left_big_result[0][3], sorted_left_big_result[0][4])
                                line_index = 0
                                for ii, [strr, x1_big, x2_big, y1_big, y2_big] in enumerate(sorted_left_big_result):
                                    if abs((y1_big + y2_big) / 2 - (team_y1 + team_y2) / 2) < 4 or left_team_x - (
                                            x1_big + x2_big) / 2 > 100:
                                        continue

                                    # 循环建立四个对象，locals()函数可以将字符串转换为变量名！
                                    # 具体的操作和含义我并不清楚，大家可以自行百度～
                                    if abs((yy1 + yy2) / 2 - (y1_big + y2_big) / 2) > 3:
                                        left_line.append(locals()['left_oneline_' + str(line_index)])
                                        del locals()['left_oneline_' + str(line_index)]
                                        line_index += 1
                                        locals()['left_oneline_' + str(line_index)] = OneLine(
                                            sorted_left_big_result[ii][3],
                                            sorted_left_big_result[ii][4])
                                        yy1 = y1_big
                                        yy2 = y2_big

                                    locals()['left_oneline_' + str(line_index)].strr += (strr + ' ')
                                left_line.append(locals()['left_oneline_' + str(line_index)])
                                del locals()['left_oneline_' + str(line_index)]

                            if right_big_temp_result:
                                sorted_right_big_result = sorted(right_big_temp_result,
                                                                 key=lambda student: (student[3], student[2]))
                                yy1 = sorted_right_big_result[0][3]
                                yy2 = sorted_right_big_result[0][4]
                                right_oneline_0 = OneLine(sorted_right_big_result[0][3], sorted_right_big_result[0][4])
                                line_index = 0
                                for ii, [strr, x1_big, x2_big, y1_big, y2_big] in enumerate(sorted_right_big_result):
                                    if abs((y1_big + y2_big) / 2 - (team_y1 + team_y2) / 2) < 4 or (
                                            x1_big + x2_big) / 2 - right_team_x > 100:
                                        continue

                                    # 循环建立四个对象，locals()函数可以将字符串转换为变量名！
                                    if abs((yy1 + yy2) / 2 - (y1_big + y2_big) / 2) > 3:
                                        right_line.append(locals()['right_oneline_' + str(line_index)])
                                        del locals()['right_oneline_' + str(line_index)]
                                        line_index += 1
                                        locals()['right_oneline_' + str(line_index)] = OneLine(
                                            sorted_right_big_result[ii][3],
                                            sorted_right_big_result[ii][4])
                                        yy1 = y1_big
                                        yy2 = y2_big

                                    locals()['right_oneline_' + str(line_index)].strr += (strr + ' ')
                                right_line.append(locals()['right_oneline_' + str(line_index)])
                                del locals()['right_oneline_' + str(line_index)]

                            player_name_dic, big_candidate, score_time_dic, Player_Time_list, big_flag = check_line(
                                left_line, 1, player_name_dic, big_candidate, score_time_dic, i, Player_Time_list,
                                big_flag)

                            player_name_dic, big_candidate, score_time_dic, Player_Time_list, big_flag = check_line(
                                right_line, -1, player_name_dic, big_candidate, score_time_dic, i,
                                Player_Time_list,
                                big_flag)

                    # 如果检测出比分，则帧数+1检测下一帧，否则帧数+200
                    if goal_flag:
                        i += 1
                        mode = 2
                        continue
                    elif big_flag:
                        i += 2
                        big_nums += 5
                        continue
                    # 没有大字幕持续若干次
                    elif big_nums < 4 and not big_flag:
                        big_nums += 1
                        i += 2
                        continue
                    # 从有大字幕突然变成没有大字幕
                    elif big_nums > 6 and not big_flag:
                        big_nums = 1
                        i += 5
                        continue
                    # 没有大字幕持续若干次后，进行跨越，再检测
                    elif big_nums == 4:
                        # if i+10000 < frame_count-5000:
                        #     i += 10000
                        # else:
                        #     i = max(i+1000, frame_count-5000)
                        if i + 5000 < frame_count - 4000:
                            i += 5000
                        elif i + 3000 < frame_count - 2000:
                            i += 3000
                        elif i > frame_count - 100:
                            i += 5
                            continue
                        else:
                            i = max(i + 2000, frame_count - 2000)
                        big_nums = 6
                        continue
                    elif i > frame_count - 100:
                        i += 5
                        continue
                    else:
                        i += 50
                        # print("can not get frames!")
                        continue

                # 把big_candidate里的值（set）取交集，如果不为空，就在player_name_dic里面看谁的key次数多
                # 次数少的key就在big_candidate中删除
                nn = len(Player_Time_list)
                # names = list(Player_Time_list.keys())
                score_times = list(score_time_dic.keys())
                for i in range(nn):
                    if Player_Time_list[i].use == 1:
                        i_score_times = list(Player_Time_list[i].goaltime.keys())
                        i_playername = Player_Time_list[i].playername

                        # 对于其他球员
                        for j in range(i + 1, nn):
                            if Player_Time_list[j].use == 1:
                                j_score_times = list(Player_Time_list[j].goaltime.keys())
                                j_playername = Player_Time_list[j].playername
                                set_temp = set(i_score_times) & set(j_score_times)
                                if set_temp:
                                    if player_name_dic[i_playername] >= player_name_dic[j_playername]:
                                        # Player_Time_list[i].goaltime.update(Player_Time_list[j].goaltime)
                                        Player_Time_list[j].use = 0
                                    else:
                                        # Player_Time_list[j].goaltime.update(Player_Time_list[i].goaltime)
                                        Player_Time_list[i].use = 0
                                        break
                                elif fuzz.token_set_ratio(i_playername, j_playername) >= 70:
                                    # 处理进球时间微调的情况，+1或者-1，需要删除前面的，而不是直接并集
                                    i_times = list(Player_Time_list[i].goaltime.keys())
                                    j_times = list(Player_Time_list[j].goaltime.keys())
                                    for j_time in j_times:
                                        if "+" in j_time:
                                            sj_index = j_time.find('+')
                                            jj_time = int(j_time[0:sj_index]) + int(j_time[sj_index:])
                                        else:
                                            jj_time = int(j_time)
                                        for i_time in i_times:
                                            if Player_Time_list[i].goaltime[i_time]:
                                                if j_time == i_time:
                                                    continue
                                                else:
                                                    if "+" in i_time:
                                                        # 格式为“90+xx”的时候
                                                        si_index = i_time.find('+')
                                                        ii_time = int(i_time[0:si_index]) + int(i_time[si_index:])
                                                    else:
                                                        ii_time = int(i_time)
                                                    if abs(jj_time - ii_time) == 1:
                                                        Player_Time_list[i].goaltime.pop(i_time)
                                    # 根据次数多少选择保留的对象
                                    if player_name_dic[i_playername] >= player_name_dic[j_playername]:
                                        Player_Time_list[i].goaltime.update(Player_Time_list[j].goaltime)
                                        Player_Time_list[j].use = 0
                                    else:
                                        Player_Time_list[j].goaltime.update(Player_Time_list[i].goaltime)
                                        Player_Time_list[i].use = 0
                                        break
                        # 与其他的比较后：
                        if Player_Time_list[i].use == 1:
                            # 如果该运动员的某个得分时间次数出现少于5，就删除这个得分时间
                            for score_time in list(Player_Time_list[i].goaltime.keys()):
                                if score_time_dic[score_time] < 5:
                                    Player_Time_list[i].goaltime.pop(score_time)
                            # 如果该球员没有得分时间，就删除该球员的记录
                            if not Player_Time_list[i].goaltime:
                                Player_Time_list[i].use = 0
                            if player_name_dic[i_playername] < 5:
                                Player_Time_list[i].use = 0

                Summary_Json = {}
                Summary_Json['teamA'] = left_team
                Summary_Json['teamB'] = right_team
                random_events = {}
                goal_event = []
                for i in range(nn):
                    if Player_Time_list[i].use:
                        name = Player_Time_list[i].playername
                        if Player_Time_list[i].location == 1:
                            team = left_team
                        else:
                            team = right_team
                        print("name:{}  team:{}".format(name, team))
                        print(Player_Time_list[i].goaltime)

                        gv = {'team': team, 'player': name, 'goal_time': Player_Time_list[i].goaltime}
                        goal_event.append(gv)

                random_events['goal_event'] = goal_event
                Summary_Json['random_events'] = random_events
                with open('event_json' + '.json', 'w') as fd:
                    json.dump(Summary_Json, fd)
                print_json(Summary_Json)

                # 把big_candidate里的值（set）取交集，如果不为空，就在player_name_dic里面看谁的key次数多
                # 次数少的key就在big_candidate中删除
                nn = len(big_candidate)
                names = list(big_candidate.keys())
                score_times = list(score_time_dic.keys())
                for i in range(nn):
                    if big_candidate[names[i]] != set('0'):
                        # 如果该运动员的某个得分时间次数出现少于5，就删除这个得分时间
                        for score_time in list(big_candidate[names[i]]):
                            if score_time_dic[score_time] < 5:
                                big_candidate[names[i]].remove(score_time)
                        # 如果该球员没有得分时间，就删除该球员的记录
                        if big_candidate[names[i]] == set():
                            big_candidate[names[i]] = set('0')
                        # 对于其他球员
                        for j in range(i + 1, nn):
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
                    elif re.findall("\d\d", key):
                        del big_candidate[key]

                print(big_candidate)
                time2 = time.time()
                print("time for this video:{}".format(time2 - time1))

                # SCORES[video_index] = score_record

                # with open('scores_3' + '.json', 'w') as fd:
                #     json.dump(SCORES, fd)

    print("total_num:{}   get:{}   miss:{}  more:{}".format(total_num, get, miss, more))


get_frames(video_dir=VIDEO_DIR)
