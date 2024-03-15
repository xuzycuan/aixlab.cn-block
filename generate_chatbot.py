# -*- coding: UTF-8 -*-
import os
import time


class ArgumentsChatBot:
    def __init__(self):
        self.SSH_SETTING = False
        self.AUD_TO_TEXT = False
        self.TEXT_TO_VEC = False
        self.GET_SIMILARITY = False
        self.EMO_MODULE = False

        self.HOST = '192.168.0.0'
        self.PORT = 22
        self.USERNAME = 'pi'
        self.PASSWORD = 'raspberry'
        self.REMOTE_PATH = '/home/pi/audio'
        self.METRIC = 'CosineSimilarity'
        self.THRESHOLD = 0.3
        self.TIRED_PROB = 0.5


def generate_chatbot(target_path, args_json):
    args_chatbot = ArgumentsChatBot()
    block_list = []
    # 解析参数
    items = args_json['algorithmList']
    for item in items:
        if item['id'] == 1015:
            block_list.append('SSH_SETTING')
            args_chatbot.SSH_SETTING = True
            args_chatbot.HOST = item['HOST']
            args_chatbot.PORT = item['PORT']
            args_chatbot.USERNAME = item['USERNAME']
            args_chatbot.PASSWORD = item['PASSWORD']
            args_chatbot.REMOTE_PATH = item['REMOTE_PATH']
        if item['id'] == 1016:
            block_list.append('AUD_TO_TEXT')
            args_chatbot.AUD_TO_TEXT = True
        if item['id'] == 1017:
            block_list.append('TEXT_TO_VEC')
            args_chatbot.TEXT_TO_VEC = True
        if item['id'] == 1018:
            block_list.append('GET_SIMILARITY')
            args_chatbot.GET_SIMILARITY = True
            args_chatbot.METRIC = item['METRIC']
            args_chatbot.THRESHOLD = item['THRESHOLD']
        if item['id'] == 1019:
            block_list.append('EMO_MODULE')
            args_chatbot.EMO_MODULE = True
            args_chatbot.TIRED_PROB = item['TIRED_PROB']

    # 写其他文件
    with open(os.path.join('Templates_chatbot', 'asr.py'), 'r', encoding='utf-8', errors='ignore') as f_template:
        lines = f_template.readlines()
        with open(os.path.join(target_path, 'asr.py'), mode='w', encoding='utf-8') as f_target:
            for line in lines:
                print(line, end='', file=f_target)
    with open(os.path.join('Templates_chatbot', '停用词.txt'), 'r', encoding='utf-8', errors='ignore') as f_template:
        lines = f_template.readlines()
        with open(os.path.join(target_path, '停用词.txt'), mode='w', encoding='utf-8') as f_target:
            for line in lines:
                print(line, end='', file=f_target)
    with open(os.path.join('Templates_chatbot', '词典.txt'), 'r', encoding='utf-8', errors='ignore') as f_template:
        lines = f_template.readlines()
        with open(os.path.join(target_path, '词典.txt'), mode='w', encoding='utf-8') as f_target:
            for line in lines:
                print(line, end='', file=f_target)

    # 按顺序写模块
    with open(os.path.join('Templates_chatbot', 'template_chatbot.py'), 'r', encoding='utf-8', errors='ignore') as f_template:
        lines = f_template.readlines()
        with open(os.path.join(target_path, 'generated.py'), mode='w', encoding='utf-8') as f_target:

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
            print('SSH_SETTING = ' + str(args_chatbot.SSH_SETTING), file=f_target)
            print('AUD_TO_TEXT = ' + str(args_chatbot.AUD_TO_TEXT), file=f_target)
            print('TEXT_TO_VEC = ' + str(args_chatbot.TEXT_TO_VEC), file=f_target)
            print('GET_SIMILARITY = ' + str(args_chatbot.GET_SIMILARITY), file=f_target)
            print('EMO_MODULE = ' + str(args_chatbot.EMO_MODULE), file=f_target)
            print(file=f_target)
            print('HOST = ' + "'" + str(args_chatbot.HOST) + "'", file=f_target)
            print('PORT = ' + str(args_chatbot.PORT), file=f_target)
            print('USERNAME = ' + "'" + str(args_chatbot.USERNAME) + "'", file=f_target)
            print('PASSWORD = ' + "'" + str(args_chatbot.PASSWORD) + "'", file=f_target)
            print('REMOTE_PATH = ' + "'" + str(args_chatbot.REMOTE_PATH) + "'", file=f_target)
            print('METRIC = ' + "'" + str(args_chatbot.METRIC) + "'", file=f_target)
            print('THRESHOLD = ' + str(args_chatbot.THRESHOLD), file=f_target)
            print('TIRED_PROB = ' + str(args_chatbot.TIRED_PROB), file=f_target)
            print('""" GENERATE Arguments END """\n', file=f_target)

            # 写各个模块
            for block in block_list:
                if block == 'SSH_SETTING':
                    for line in lines:
                        if 'GENERATE SSHSetting START' in line:
                            flag = True
                        if flag:
                            print(line, end='', file=f_target)
                        if 'GENERATE SSHSetting END' in line:
                            flag = False
                    print(file=f_target)
                if block == 'AUD_TO_TEXT':
                    for line in lines:
                        if 'GENERATE AudToText START' in line:
                            flag = True
                        if flag:
                            print(line, end='', file=f_target)
                        if 'GENERATE AudToText END' in line:
                            flag = False
                    print(file=f_target)
                if block == 'TEXT_TO_VEC':
                    for line in lines:
                        if 'GENERATE TextToVec START' in line:
                            flag = True
                        if flag:
                            print(line, end='', file=f_target)
                        if 'GENERATE TextToVec END' in line:
                            flag = False
                    print(file=f_target)
                if block == 'GET_SIMILARITY':
                    for line in lines:
                        if 'GENERATE GetSimilarity START' in line:
                            flag = True
                        if flag:
                            print(line, end='', file=f_target)
                        if 'GENERATE GetSimilarity END' in line:
                            flag = False
                    print(file=f_target)
                if block == 'EMO_MODULE':
                    for line in lines:
                        if 'GENERATE EmoModule START' in line:
                            flag = True
                        if flag:
                            print(line, end='', file=f_target)
                        if 'GENERATE EmoModule END' in line:
                            flag = False
                    print(file=f_target)

            # 写主函数（结尾）
            for line in lines:
                if 'GENERATE Main START' in line:
                    flag = True
                if flag:
                    print(line, end='', file=f_target)
                if 'GENERATE Main END' in line:
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
    generate_chatbot('output', a)
