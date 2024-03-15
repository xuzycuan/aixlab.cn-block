# -*- coding: UTF-8 -*-
import os
import json
import sys
import time
from threading import Thread
from generate_json import generate_json
from generate_ml import generate_ml
from generate_ga import generate_ga
from generate_chatbot import generate_chatbot 


def generate_o(file_path):
    with open(os.path.join(file_path, 'generated.json'), mode='w', encoding='utf-8') as f:
        print('test', file=f)
    return os.path.join(file_path, 'generated.json')


def generate(path, template_type, json_argument):
    # 参数文件转换成json
    a = json.loads(json_argument)

    # 判断生成文件模板
    if (int(template_type)) == 1:
        generate_json(path, a)
        generate_ml(path,a)
    if (int(template_type)) == 2:
        end = path.rindex("/")
        path = path[0:end]
        generate_ga(path, a)
    if (int(template_type)) == 3:
        end = path.rindex("/")
        path = path[0:end]
        generate_chatbot(path, a)


def train_py(path_py, path_folder):
    from importlib import reload
    sys.path.append(path_py)
    import generated
    from generated import train
    reload(generated)
    result = train(path_folder, os.path.join(path_py, 'Result'))
    sys.path.remove(path_py)
    return result


def run_py(path_py, path_file):
    from importlib import reload
    sys.path.append(path_py)
    import generated
    from generated import run
    reload(generated)
    result = run(path_file, os.path.join(path_py, 'Result'))
    sys.path.remove(path_py)
    return result

def val_py(path_py, path_file):
    from importlib import reload
    sys.path.append(path_py)
    import generated
    from generated import val
    reload(generated)
    result = val(path_file, os.path.join(path_py, 'Result'))
    sys.path.remove(path_py)
    return result


def run_py2(path_py):
    print("====" + path_py + "=====" )
    #from importlib import reload
    #sys.path.append(path_py)
    #import play_gobang
    #from play_gobang import run
    #reload(play_gobang)

    thr = Thread(target=run_game_algorithm, args=(path_py,))
    thr.start()
    result = "success"
    #sys.path.remove(path_py)
    return result


def run_game_algorithm(path_py):
    import subprocess
    #py_to_run = os.path.join(path_py, 'play=====.py')
    p = subprocess.Popen('python ' + path_py, stdout=subprocess.PIPE, shell=True)
    print(p.stdout.readlines())


def asyncfun(f):
    def wrapper(*args, **kwargs):
        thr = Thread(target=f, args=args, kwargs=kwargs)
        thr.start()
    return wrapper
def runpy(path_py):
    os.system("python " + path_py + "generated.py")

if __name__ == '__main__':

    # # 读参数文件
    # with open('args.txt', 'r') as f:
    #     lines = f.readlines()
    #     a = ''
    #     for line in lines:
    #         a += line.strip()
    # a = a.split()
    # a = ''.join(a)

    # generate('output', 2, a)

    run_py2('output')
