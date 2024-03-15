# -*- coding: UTF-8 -*-
""" 2022.03.29 - 14:10:26 """
""" GENERATE Import START """
import sys
from math import *
import numpy as np
import pdb
""" GENERATE Import END """

""" GENERATE Arguments START """
SET_BOARD = True
EVAL_FUN = True
ALPHABETA = True
COLUMN = 10
ROW = 10
DEPTH = 3
RATIO = 1
PT_01100 = 1
PT_00110 = 1
PT_11010 = 4
PT_01011 = 4
PT_00111 = 10
PT_11100 = 10
PT_01110 = 100
PT_010110 = 100
PT_011010 = 100
PT_11101 = 100
PT_11011 = 100
PT_10111 = 100
PT_11110 = 100
PT_01111 = 100
PT_011110 = 1000
PT_11111 = 90000
""" GENERATE Arguments END """

""" GENERATE SetBoard START """
list1 = []  # AI
list2 = []  # human
list3 = []  # all

list_all = []  # 整个棋盘的点
next_point = [0, 0]  # AI下一步最应该下的位置
for i in range(COLUMN+1):
    for j in range(ROW+1):
        list_all.append((i, j))
""" GENERATE SetBoard END """

""" GENERATE EvalFun START """
# 棋型的评估分数
shape_score = [(PT_01100, (0, 1, 1, 0, 0)),#活2
            (PT_00110, (0, 0, 1, 1, 0)),#活2
            (PT_11010, (1, 1, 0, 1, 0)),#眠3
            (PT_01011, (0, 1, 0, 1, 1)),#眠3
            (PT_00111, (0, 0, 1, 1, 1)),#眠3
            (PT_11100, (1, 1, 1, 0, 0)),#眠3
            (PT_01110, (0, 1, 1, 1, 0)),#活3
            (PT_010110, (0, 1, 0, 1, 1, 0)),#活3
            (PT_011010, (0, 1, 1, 0, 1, 0)),#活3
            (PT_11101, (1, 1, 1, 0, 1)),#冲4
            (PT_11011, (1, 1, 0, 1, 1)),#冲4
            (PT_10111, (1, 0, 1, 1, 1)),#冲4
            (PT_11110, (1, 1, 1, 1, 0)),#冲4
            (PT_01111, (0, 1, 1, 1, 1)),#冲4
            (PT_011110, (0, 1, 1, 1, 1, 0)),#活4
            (PT_11111, (1, 1, 1, 1, 1))]#连5


# 评估函数
def evaluation(is_ai):
    total_score = 0

    if is_ai:
        my_list = list1
        enemy_list = list2
    else:
        my_list = list2
        enemy_list = list1

    # 算自己的得分
    score_all_arr = []  # 得分形状的位置 用于计算如果有相交 得分翻倍
    my_score = 0
    for pt in my_list:
        m = pt[0]
        n = pt[1]
        my_score += cal_score(m, n, 0, 1, enemy_list, my_list, score_all_arr)
        my_score += cal_score(m, n, 1, 0, enemy_list, my_list, score_all_arr)
        my_score += cal_score(m, n, 1, 1, enemy_list, my_list, score_all_arr)
        my_score += cal_score(m, n, -1, 1, enemy_list, my_list, score_all_arr)

    #  算敌人的得分， 并减去
    score_all_arr_enemy = []
    enemy_score = 0
    for pt in enemy_list:
        m = pt[0]
        n = pt[1]
        enemy_score += cal_score(m, n, 0, 1, my_list, enemy_list, score_all_arr_enemy)
        enemy_score += cal_score(m, n, 1, 0, my_list, enemy_list, score_all_arr_enemy)
        enemy_score += cal_score(m, n, 1, 1, my_list, enemy_list, score_all_arr_enemy)
        enemy_score += cal_score(m, n, -1, 1, my_list, enemy_list, score_all_arr_enemy)

    total_score = my_score - enemy_score*RATIO*0.1

    return total_score


# 每个方向上的分值计算
def cal_score(m, n, x_decrict, y_derice, enemy_list, my_list, score_all_arr):
    add_score = 0  # 加分项
    # 在一个方向上， 只取最大的得分项
    max_score_shape = (0, None)

    # 如果此方向上，该点已经有得分形状，不重复计算
    for item in score_all_arr:
        for pt in item[1]:
            if m == pt[0] and n == pt[1] and x_decrict == item[2][0] and y_derice == item[2][1]:
                return 0

    # 在落子点 左右方向上循环查找得分形状
    for offset in range(-5, 1):
        # offset = -2
        pos = []
        for i in range(0, 6):
            if (m + (i + offset) * x_decrict, n + (i + offset) * y_derice) in enemy_list:
                pos.append(2)
            elif (m + (i + offset) * x_decrict, n + (i + offset) * y_derice) in my_list:
                pos.append(1)
            else:
                pos.append(0)
        tmp_shap5 = (pos[0], pos[1], pos[2], pos[3], pos[4])
        tmp_shap6 = (pos[0], pos[1], pos[2], pos[3], pos[4], pos[5])

        for (score, shape) in shape_score:
            if tmp_shap5 == shape or tmp_shap6 == shape:
                if tmp_shap5 == (1,1,1,1,1):
                    print('wwwwwwwwwwwwwwwwwwwwwwwwwww')
                if score > max_score_shape[0]:
                    max_score_shape = (score, ((m + (0+offset) * x_decrict, n + (0+offset) * y_derice),
                                               (m + (1+offset) * x_decrict, n + (1+offset) * y_derice),
                                               (m + (2+offset) * x_decrict, n + (2+offset) * y_derice),
                                               (m + (3+offset) * x_decrict, n + (3+offset) * y_derice),
                                               (m + (4+offset) * x_decrict, n + (4+offset) * y_derice)), (x_decrict, y_derice))

    # 计算两个形状相交， 如两个3活 相交， 得分增加 一个子的除外
    if max_score_shape[1] is not None:
        for item in score_all_arr:
            for pt1 in item[1]:
                for pt2 in max_score_shape[1]:
                    if pt1 == pt2 and max_score_shape[0] > 10 and item[0] > 10:
                        add_score += item[0] + max_score_shape[0]

        score_all_arr.append(max_score_shape)

    return add_score + max_score_shape[0]


""" GENERATE EvalFun END """

""" GENERATE AlphaBeta START """


def ai():
    global cut_count   # 统计剪枝次数
    cut_count = 0
    global search_count   # 统计搜索次数
    search_count = 0
    negamax(True, DEPTH, -99999999, 99999999)
    print("本次共剪枝次数：" + str(cut_count))
    print("本次共搜索次数：" + str(search_count))
    return next_point[0], next_point[1]


# 负值极大算法搜索 alpha + beta剪枝
def negamax(is_ai, depth, alpha, beta):
    # print('alpha', alpha, 'beta', beta)
    # 游戏是否结束 | | 探索的递归深度是否到边界
    if game_win(list1) or game_win(list2) or depth == 0:
        return evaluation(is_ai)

    blank_list = list(set(list_all).difference(set(list3)))
    order(blank_list)   # 搜索顺序排序  提高剪枝效率
    # 遍历每一个候选步
    for next_step in blank_list:

        global search_count
        search_count += 1

        # 如果要评估的位置没有相邻的子， 则不去评估  减少计算
        if not has_neightnor(next_step):
            continue

        if is_ai:
            list1.append(next_step)
        else:
            list2.append(next_step)
        list3.append(next_step)

        value = -negamax(not is_ai, depth - 1, -beta, -alpha)
        if is_ai:
            list1.remove(next_step)
        else:
            list2.remove(next_step)
        list3.remove(next_step)

        if value > alpha:
            # print("value:" +str(value) + ">alpha:" + str(alpha) + ",beta:" + str(beta))
            # print(list3)
            if depth == DEPTH:
                next_point[0] = next_step[0]
                next_point[1] = next_step[1]
            # alpha + beta剪枝点value
            if value >= beta:
                global cut_count
                cut_count += 1
                print('第{}次剪枝:  value={}, alpha={}, beta={}'.format(cut_count,value,alpha,beta))
                return beta
            alpha = value

    return alpha


#  离最后落子的邻居位置最有可能是最优点
def order(blank_list):
    last_pt = list3[-1]
    for item in blank_list:
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                if (last_pt[0] + i, last_pt[1] + j) in blank_list:
                    blank_list.remove((last_pt[0] + i, last_pt[1] + j))
                    blank_list.insert(0, (last_pt[0] + i, last_pt[1] + j))


def has_neightnor(pt):
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                continue
            if (pt[0] + i, pt[1]+j) in list3:
                return True
    return False

# 判断最近下棋的一方是否赢了
def game_win(list):
    for m in range(COLUMN):
        for n in range(ROW):

            if n < ROW - 4 and (m, n) in list and (m, n + 1) in list and (m, n + 2) in list and (
                    m, n + 3) in list and (m, n + 4) in list:
                return True
            elif m < ROW - 4 and (m, n) in list and (m + 1, n) in list and (m + 2, n) in list and (
                        m + 3, n) in list and (m + 4, n) in list:
                return True
            elif m < ROW - 4 and n < ROW - 4 and (m, n) in list and (m + 1, n + 1) in list and (
                        m + 2, n + 2) in list and (m + 3, n + 3) in list and (m + 4, n + 4) in list:
                return True
            elif m < ROW - 4 and n > 3 and (m, n) in list and (m + 1, n - 1) in list and (
                        m + 2, n - 2) in list and (m + 3, n - 3) in list and (m + 4, n - 4) in list:
                return True
    return False


""" GENERATE AlphaBeta END """

""" GENERATE GoBangClass START """


class GoBang():
    def __init__(self,AI_first=False):
        self.g = 0
        self.change = 0
        self.AI_first=AI_first

    def humanRun(self, x1, y1):
        global list1  # AI
        global list2  # human
        global list3  # all
        global list_all  # 整个棋盘的点

        p2 = (x1, y1)
        if ((p2[0], p2[1]) in list3):
            # self.g = 4
            raise ValueError("这个位置已经有棋子啦")             
              
        elif ((p2[0], p2[1]) not in list_all):
            # self.g = 4
            raise ValueError("不可用的位置")
        else:
            a2 = p2[0]
            b2 = p2[1]
            list2.append((a2, b2))  # 把人下过的位置记录到list1中
            list3.append((a2, b2))  # list3记录所有已经有子的位置坐标                    
            print('\n第'+str(self.change)+'步,人类落子位置：'+str((a2,b2)))
            self.change += 1

        if game_win(list2):
            self.g = 2
            
            
    def AIRun(self):
        global list1  # AI
        global list2  # human
        global list3  # all
        global list_all  # 整个棋盘的点
        # 第0/2/4步，人类下完之后没有获胜，则轮到AI下棋
        print('\n第'+str(self.change)+'步,AI开始搜索：')
        self.pos = ai()  # ai算出一个下棋的位置坐标(x,y)
        # print('第'+str(self.change)+'步:')
        print('第'+str(self.change)+'步搜索结束,AI落子位置：'+str(self.pos)+"\n")
        if self.pos in list3:  # 如果无路可走，棋局结束
            self.g = 4  # g=4表示无路可走
            return

        list1.append(self.pos)  # 把AI下过的位置记录到list1中
        list3.append(self.pos)  # list3记录所有已经有子的位置坐标
        self.change += 1

        if game_win(list1):
            self.g = 1             

        return self.pos

    def AI_first_Run(self): 
        global list1 
        global list2 
        global list3  
        list1.append((7,7))
        list3.append((7,7))
        print('\n第'+str(self.change)+'步:')
        print('AI落子位置：',(7,7))
        self.change += 1 
        
        return (7,7)

    def initialize(self):
        global list1 
        global list2 
        global list3
        global next_point
        self.g= 0
        self.change = 0
        list1 = []  # AI
        list2 = []  # human
        list3 = []  # all
        next_point = [0, 0]  # AI下一步最应该下的位置


""" GENERATE GoBangClass END """

