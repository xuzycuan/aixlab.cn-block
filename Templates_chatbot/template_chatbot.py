# coding=utf-8
""" GENERATE Import START """
import os
import csv
import time
import random
import paramiko
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import jieba

from asr import asr

""" GENERATE Import END """

""" GENERATE Arguments START """
SSH_SETTING = True
AUD_TO_TEXT = True
TEXT_TO_VEC = True
GET_SIMILARITY = True
EMO_MODULE = True

HOST = '192.168.0.0'
PORT = 22
USERNAME = 'pi'
PASSWORD = 'raspberry'
REMOTE_PATH = '/home/pi/audio'
METRIC = 'CosineSimilarity'
THRESHOLD = 0.3
TIRED_PROB = 0.5
""" GENERATE Arguments END """


""" GENERATE SSHSetting START """


# Raspi录音5秒
def record():
    cmd = 'arecord -Dhw:2,0 -d 4 -f cd -r 8000 -c 1 -t wav ' + REMOTE_PATH + '/rec.wav'
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(HOST, PORT, USERNAME, PASSWORD)
    stdin, stdout, stderr = ssh_client.exec_command(cmd)
    with open('temp.txt', 'w') as f:
        print(stdout.read().decode(), file=f)
        print(stderr.read().decode(), file=f)
    ssh_client.close()


# 下载服务器的文件到指定目录
def download():
    local_path = 'rec.wav'
    remote_path = REMOTE_PATH + '/rec.wav'
    transport = paramiko.Transport((HOST, 22))
    transport.connect(username=USERNAME, password=PASSWORD)
    sftp = paramiko.SFTPClient.from_transport(transport)
    sftp.get(remote_path, local_path)
    sftp.close()


# 播放树莓派的回答
def play(ans):
    cmd = 'aplay /home/pi/audio/' + str(ans) + '.wav'
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(HOST, PORT, USERNAME, PASSWORD)
    stdin, stdout, stderr = ssh_client.exec_command(cmd)
    with open('temp.txt', 'w') as f:
        print(stdout.read().decode(), file=f)
        print(stderr.read().decode(), file=f)
    ssh_client.close()


""" GENERATE SSHSetting END """

""" GENERATE AudToText START """
# import部分载入asr方法
""" GENERATE AudToText END """

""" GENERATE TextToVec START """


# 读QA列表
def read_corpus(file_path):
    qa_file = []
    with open(file_path, 'r', encoding='gb2312') as f:
        reader = csv.DictReader(f)
        for row in reader:
            qa_file.append(row)
    qlist = []  # 问题列表
    alist = []  # 答案列表
    for i in range(len(qa_file)):
        qlist.append(qa_file[i]['question'])
        alist.append(qa_file[i]['answer'])
    return qlist, alist


# 分词
def cut(sentence):
    jieba.load_userdict('词典.txt')
    results = []
    for sen in sentence:
        result = []
        for word in jieba.cut(sen):
            result.append(word)
        results.append(result)

    return results


# 预处理
def text_preprocessing(input_list):
    stop_words = [line.strip() for line in open('停用词.txt', 'r', encoding='UTF-8').readlines()]  # stopwordslist
    input_list = cut(input_list)  # 分词
    new_list = []  # 保存处理完的qlist\alist

    for l in input_list:
        l_list = ''  # 保存句子
        for word in l:
            word = ''.join(c for c in word if c not in string.punctuation)  # 去除所有标点符号

            if word not in stop_words:  # 5.去停用词
                l_list += word + ' '
        new_list.append(l_list)
    return new_list


""" GENERATE TextToVec END """

""" GENERATE GetSimilarity START """


def get_similarity(query_vec, X):
    if METRIC == 'CosineSimilarity':
        return cosine_similarity(query_vec, X)[0]  # 计算query跟每个库里的问题之间的相似度


""" GENERATE GetSimilarity END """

""" GENERATE EmoModule START """
# Run方法中直接使用情感方法
""" GENERATE EmoModule END """

""" GENERATE Main START """


def run(qa_path):
    # 提前计算好预设问题的向量
    qlist, alist = read_corpus(qa_path)
    qlist_d = text_preprocessing(qlist)   # 预处理后的问题列表
    vectorizer = TfidfVectorizer()          # 定一个tf-idf的vectorizer
    X = vectorizer.fit_transform(qlist_d)  # 结果存放在X矩阵
    alist = np.array(alist)

    while True:
        cmd_input = input('1：对话   0：结束\n')
        if cmd_input == '0':
            break
        elif cmd_input != '1':
            printsmiley('请重新选择。 1：对话   0：结束')
        elif cmd_input == '1':
            printsmiley('录音中...')
            # record()
            printsmiley('请稍等，我正在思考！！')
            time.sleep(1)
            # download()
            text = '测试'
            # text = asr('rec.wav')  # 得到音频对应的文字

            input_q = text_preprocessing([text])  # 对输入的query问题进行预处理
            input_vec = vectorizer.transform(input_q)  # 转为tfidf向量
            similarity_res = get_similarity(input_vec, X)  # 计算query跟每个库里的问题之间的相似度
            top_idxs = np.argsort(similarity_res)[-1:].tolist()  # top_idxs存放相似度最高的（存在qlist里的）问题的下标 np.argsort输出排序后的下标

            if similarity_res[top_idxs[-1]] < THRESHOLD:
                ans = '请重新提问'
            else:
                ans = alist[top_idxs[-1]]

            if EMO_MODULE:
                if ans == 'music':
                    x = random.randint(0, 9)
                    if x < (10 * TIRED_PROB):
                        ans = 'tired'

            printsmiley(ans)
            # play(ans)


def printsmiley(text):
    print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n')
    print(r'            _-""""""-_    ')
    print(r"          .':::::::::::.  ")
    print(r'         /::::::::::::::\ ')
    print(r'        |:::(o)::::(o):::|')
    print(r'        |::::::::::::::::|')
    print(r'        |:::\::::::::/:::|')
    print(r"         \:::`.____.':::/ ")
    print(r"          `.::::::::::.'  ")
    print(r"            ``-....-''    ")
    print('\n\n\n')
    print('========================')
    print(text)
    print('========================')


if __name__ == '__main__':
    run('../input/final.csv')

""" GENERATE Main END """
