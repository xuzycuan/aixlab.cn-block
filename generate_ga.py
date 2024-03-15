# -*- coding: UTF-8 -*-
import os
import time


class ArgumentsGA:
    def __init__(self):
        self.SET_BOARD = False
        self.EVAL_FUN = False
        self.ALPHABETA = False

        self.COLUMN = 10
        self.ROW = 10
        self.DEPTH = 3
        self.RATIO = 1

        self.PT_01100 = 1
        self.PT_00110 = 1
        self.PT_11010 = 4
        self.PT_01011 = 4
        self.PT_00111 = 10
        self.PT_11100 = 10
        self.PT_01110 = 100
        self.PT_010110 = 100
        self.PT_011010 = 100
        self.PT_11101 = 100
        self.PT_11011 = 100
        self.PT_10111 = 100
        self.PT_11110 = 100
        self.PT_01111 = 100
        self.PT_011110 = 1000
        self.PT_11111 = 90000


def generate_ga(target_path, args_json):
    # print(target_path)
    # print(args_json)
    args_ga = ArgumentsGA()
    block_list = []

    # 解析参数
    items = args_json['algorithmList']
    for item in items:
        if item['id'] == 1012:
            block_list.append('SET_BOARD')
            args_ga.SET_BOARD = True
            args_ga.COLUMN = item['COLUMN']
            args_ga.ROW = item['ROW']
        if item['id'] == 1002:
            block_list.append('EVAL_FUN')
            args_ga.ALPHABETA = True
            args_ga.DEPTH = item['DEPTH']
        if item['id'] == 1001:
            block_list.append('ALPHABETA')
            args_ga.EVAL_FUN = True
            args_ga.RATIO = item['RATIO']
            args_ga.PT_01100 = item['PT_01100']
            args_ga.PT_00110 = item['PT_00110']
            args_ga.PT_11010 = item['PT_11010']
            args_ga.PT_01011 = item['PT_01011']
            args_ga.PT_00111 = item['PT_00111']
            args_ga.PT_11100 = item['PT_11100']
            args_ga.PT_01110 = item['PT_01110']
            args_ga.PT_010110 = item['PT_010110']
            args_ga.PT_011010 = item['PT_011010']
            args_ga.PT_11101 = item['PT_11101']
            args_ga.PT_11011 = item['PT_11011']
            args_ga.PT_10111 = item['PT_10111']
            args_ga.PT_11110 = item['PT_11110']
            args_ga.PT_01111 = item['PT_01111']
            args_ga.PT_011110 = item['PT_011110']
            args_ga.PT_11111 = item['PT_11111']

    # 写Graphics文件
    with open(os.path.join('Templates_ga', 'graphics.py'), 'r', encoding='utf-8', errors='ignore') as f_template:
        lines = f_template.readlines()
        with open(os.path.join(target_path, 'graphics.py'), mode='w', encoding='utf-8') as f_target:
            for line in lines:
                print(line, end='', file=f_target)

    # 写Play文件
    with open(os.path.join('Templates_ga', 'play_gobang.py'), 'r', encoding='utf-8', errors='ignore') as f_template:
        lines = f_template.readlines()
        with open(os.path.join(target_path, 'play_gobang.py'), mode='w', encoding='utf-8') as f_target:
            for line in lines:
                print(line, end='', file=f_target)

    # 按顺序写模块
    with open(os.path.join('Templates_ga', 'gobang_class.py'), 'r', encoding='utf-8', errors='ignore') as f_template:
        lines = f_template.readlines()
        with open(os.path.join(target_path, 'gobang_class.py'), mode='w', encoding='utf-8') as f_target:

            # 写开头
            flag = False
            print('# -*- coding: UTF-8 -*-', file=f_target)
            print('"""', time.strftime('%Y.%m.%d - %H:%M:%S', time.localtime()), '"""', file=f_target)
            for line in lines:
                if 'GENERATE Import START' in line:
                    flag = True
                if flag:
                    print(line, end='', file=f_target)
                if 'GENERATE Import END' in line:
                    flag = False
            print(file=f_target)

            # 写参数
            print('""" GENERATE Arguments START """', file=f_target)
            print('SET_BOARD = ' + str(args_ga.SET_BOARD), file=f_target)
            print('EVAL_FUN = ' + str(args_ga.EVAL_FUN), file=f_target)
            print('ALPHABETA = ' + str(args_ga.ALPHABETA), file=f_target)
            print('COLUMN = ' + str(args_ga.COLUMN), file=f_target)
            print('ROW = ' + str(args_ga.ROW), file=f_target)
            print('DEPTH = ' + str(args_ga.DEPTH), file=f_target)
            print('RATIO = ' + str(args_ga.RATIO), file=f_target)
            print('PT_01100 = ' + str(args_ga.PT_01100), file=f_target)
            print('PT_00110 = ' + str(args_ga.PT_00110), file=f_target)
            print('PT_11010 = ' + str(args_ga.PT_11010), file=f_target)
            print('PT_01011 = ' + str(args_ga.PT_01011), file=f_target)
            print('PT_00111 = ' + str(args_ga.PT_00111), file=f_target)
            print('PT_11100 = ' + str(args_ga.PT_11100), file=f_target)
            print('PT_01110 = ' + str(args_ga.PT_01110), file=f_target)
            print('PT_010110 = ' + str(args_ga.PT_010110), file=f_target)
            print('PT_011010 = ' + str(args_ga.PT_011010), file=f_target)
            print('PT_11101 = ' + str(args_ga.PT_11101), file=f_target)
            print('PT_11011 = ' + str(args_ga.PT_11011), file=f_target)
            print('PT_10111 = ' + str(args_ga.PT_10111), file=f_target)
            print('PT_11110 = ' + str(args_ga.PT_11110), file=f_target)
            print('PT_01111 = ' + str(args_ga.PT_01111), file=f_target)
            print('PT_011110 = ' + str(args_ga.PT_011110), file=f_target)
            print('PT_11111 = ' + str(args_ga.PT_11111), file=f_target)
            print('""" GENERATE Arguments END """\n', file=f_target)

            # 写各个模块
            for block in block_list:
                if block == 'SET_BOARD':
                    for line in lines:
                        if 'GENERATE SetBoard START' in line:
                            flag = True
                        if flag:
                            print(line, end='', file=f_target)
                        if 'GENERATE SetBoard END' in line:
                            flag = False
                    print(file=f_target)
                if block == 'EVAL_FUN':
                    for line in lines:
                        if 'GENERATE EvalFun START' in line:
                            flag = True
                        if flag:
                            print(line, end='', file=f_target)
                        if 'GENERATE EvalFun END' in line:
                            flag = False
                    print(file=f_target)
                if block == 'ALPHABETA':
                    for line in lines:
                        if 'GENERATE AlphaBeta START' in line:
                            flag = True
                        if flag:
                            print(line, end='', file=f_target)
                        if 'GENERATE AlphaBeta END' in line:
                            flag = False
                    print(file=f_target)

            # 写主函数（结尾）
            for line in lines:
                if 'GENERATE GoBangClass START' in line:
                    flag = True
                if flag:
                    print(line, end='', file=f_target)
                if 'GENERATE GoBangClass END' in line:
                    flag = False
            print(file=f_target)


if __name__ == '__main__':
    import json
    with open('record.json', 'r') as f:
        args_lines = f.readlines()
        a = ''
        for args_line in args_lines:
            a += args_line.strip()
        a = a.split()
        a = ''.join(a)

    a = json.loads(a)
    generate_ga('output', a)
