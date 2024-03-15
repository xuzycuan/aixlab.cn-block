from flask import Flask
from flask import request
from flask_cors import CORS
from operate import *
import json
import requests
import time
import re
import threading
import zipfile
import traceback

from tunnel_client import *
from get_face_imgs import face_detect_and_cut
from contextlib import closing

app = Flask(__name__)

tunnel = TunnelClient()


class TunnelEventProxy():
    def __init__(self):
        # self.work_path = os.path.dirname(os.path.realpath(__file__))
        self.work_path = '.'
        self.username = ''
        self.datasets_api = ''
        self.model_api = ''
        self.stop_or_not = [False]

    def download_datasets(self, datasetsId, dataFile):
        try:
            dataUrl = self.datasets_api.replace('{id}', datasetsId)

            response = requests.get(dataUrl, stream=True)

            with closing(open(dataFile, "wb")) as f:
                for chunk in response.iter_content(chunk_size=4096):
                    if chunk:
                        f.write(chunk)

            return True

        except:
            traceback.print_exc()

        return False

    def prepare_datasets(self, pyPath, cmd, projectType, localPath, datasetsId):
        if datasetsId == '0':
            if localPath == '':
                return False, localPath

            else:
                return True, localPath

        origName = ''
        dataName = ''

        if cmd == 'TRAIN':
            origName = 'train_orig'
            dataName = 'train_datas'

        elif cmd == 'RUN1':
            origName = 'run_orig'
            dataName = 'run_datas'

        elif cmd == 'VAL':
            origName = 'val_orig'
            dataName = 'val_datas'

        dataFile = os.path.join(pyPath, '%s.zip' %dataName)
        origPath = os.path.join(pyPath, origName)
        dataPath = os.path.join(pyPath, dataName)

        if dataName == '':
            return False, dataPath

        if os.path.exists(dataFile) and os.path.isfile(dataFile):
            os.remove(dataFile)

        if os.path.exists(origPath) and os.path.isdir(origPath):
            self.cleardir(origPath)

        else:
            os.mkdir(origPath)

        if os.path.exists(dataPath) and os.path.isdir(dataPath):
            self.cleardir(dataPath)

        else:
            os.mkdir(dataPath)

        if not self.download_datasets(datasetsId, dataFile):
            return False, dataPath

        if not self.unzip_file(origPath, dataFile):
            return False, dataPath

        self.prepare_face_datasets(origPath, dataPath)

        if cmd == 'RUN1' and int(projectType) == 1:
            for filename in os.listdir(dataPath):
                dataPath = os.path.join(dataPath, filename)
                break
        print(dataPath)
        return True, dataPath

    def upload_model(self, pyPath, dataPath, resultPath, projectId):
        filename = os.path.join(pyPath, "model.zip")

        if self.zip_model(pyPath, resultPath, filename):
            response_text = None

            try:
                modelUrl = self.model_api.replace('{id}', projectId)
                modelData = { 'file': ("model.zip", open(filename, 'rb'), 'application/x-zip-compressed') }
                response = requests.post(url=modelUrl, files=modelData)

                response.raise_for_status()

            except requests.RequestException as e:
                traceback.print_exc()

            else:
                if response.status_code == 200:
                    response_text = response.text

            print(response_text)

            try:
                if response_text:
                    result = json.loads(response_text)

                    if 'state' in result:
                        if typeof(result['state']) == 'int':
                            return (result['state'] == 100)

            except Exception as e:
                traceback.print_exc()

        return False

    def keepalive(self, tunnel):
        tunnel.send('HEARTBEATS', json.dumps({
            'username': self.username
        }))

        threading.Timer(30, TunnelEventProxy.keepalive, args=(self, tunnel)).start()

    def run_task(self, cmd, params):
        if cmd == 'TRAIN':
            projectType = params['projectType']
            projectId = params['projectId']
            pyPath = os.path.join(self.work_path, projectId)
            pyPathRemote = params['pyPath']
            folderPath = params['folderPath']
            datasetsId = params['datasetsId']
            rsPath = os.path.join(pyPath, 'Result')
            rsPathRemote = os.path.join(pyPathRemote, 'Result')

            success, folderPath = self.prepare_datasets(pyPath, cmd, projectType, folderPath, datasetsId)

            if success:
                # train_flag, message = self.operate_this(pyPath, folderPath, "train")
                try:
                    self.stop_or_not[0] = False
                    
                    message = train(folderPath ,os.path.join(pyPath, 'Result'), self.stop_or_not)
                    print(message)
                    if message == -1:
                        return
                    # if train_flag == 1:
                    tunnel.send_file('TRAIN:RESULT', json.dumps({
                        "projectId": projectId,
                        "result": "success",
                        "msg":str(message),
                        "resultPath": os.path.join(rsPathRemote, "result.jpg")
                    }), os.path.join(rsPath, "result.jpg"))

                    self.upload_model(pyPath, folderPath, rsPath, projectId)

                # elif train_flag == 0:
                except:
                    message = traceback.format_exc()
                    print("训练出错啦！\nError Message: " + message)

                    tunnel.send_file('TRAIN:RESULT', json.dumps({
                        "projectId": projectId,
                        "result": "fail",
                        "msg":message,
                        "resultPath": os.path.join(rsPathRemote, "result.jpg")
                    }), os.path.join(rsPath, "result.jpg"))
            
        elif cmd == 'RUN1':
            projectType = params['projectType']
            projectId = params['projectId']
            pyPath = os.path.join(self.work_path, projectId)
            pyPathRemote = params['pyPath']
            filePath = params['filePath']
            datasetsId = params['datasetsId']
            rsPath = os.path.join(pyPath, 'Result')
            rsPathRemote = os.path.join(pyPathRemote, 'Result')

            success, filePath = self.prepare_datasets(pyPath, cmd, projectType, filePath, datasetsId)

            if success:
                try:
                    msg = run(filePath, os.path.join(pyPath, 'Result'))
                # run_flag,msg = self.operate_this(pyPath, filePath, "run")

                # if run_flag == 1:
                    tunnel.send_file('RUN1:RESULT', json.dumps({
                        "projectId": projectId,
                        "result": "success",
                        "msg": msg,
                        "resultPath": os.path.join(rsPathRemote, "result.txt")
                    }), os.path.join(rsPath, "result.txt"))

                # elif run_flag == 0:
                except Exception as msg:
                    print("运行出错啦！\nError Message: " + str(msg))
            
                    tunnel.send_file('RUN1:RESULT', json.dumps({
                        "projectId": projectId,
                        "result": "fail",
                        "msg": str(msg),
                        "resultPath": os.path.join(rsPathRemote, "result.txt")
                    }), os.path.join(rsPath, "result.txt"))

        elif cmd == 'RUN2':
            projectId = params['projectId']
            pyPath = os.path.join(self.work_path, projectId)
            pyPathRemote = params['pyPath']

            # run_py2(os.path.join(pyPath, "play_gobang.py"))

            tunnel.send('RUN2:RESULT', json.dumps({
                "projectId": projectId,
                "result": "success"
            }))

        elif cmd == 'VAL':
            projectType = params['projectType']
            projectId = params['projectId']
            pyPath = os.path.join(self.work_path, projectId)
            pyPathRemote = params['pyPath']
            filePath = params['filePath']
            datasetsId = params['datasetsId']
            rsPath = os.path.join(pyPath, 'Result')
            rsPathRemote = os.path.join(pyPathRemote, 'Result')
            

            success, filePath = self.prepare_datasets(pyPath, cmd, projectType, filePath, datasetsId)

            if success:
                # val_flag, msg = self.operate_this(pyPath, filePath, "val")
                try:
                    msg = val(filePath,os.path.join(pyPath, 'Result'))
                # if val_flag == 1:
                    tunnel.send_file('VAL:RESULT', json.dumps({
                        "projectId": projectId,
                        "result": "success",
                        "msg": msg,
                        "resultPath": os.path.join(rsPathRemote, "val_result.txt")
                    }), os.path.join(rsPath, "val_result.txt"))
                
                except Exception as msg:
                # elif val_flag == 0:
                    print("验证出错啦！\nError Message: " + str(msg))
                    tunnel.send_file('VAL:RESULT', json.dumps({
                        "projectId": projectId,
                        "result": "fail",
                        "msg": str(msg),
                        "resultPath": os.path.join(rsPathRemote, "val_result.txt")
                    }), os.path.join(rsPath, "val_result.txt"))

    def run(self, tunnel, action, data, filedata):
        if action == b'TRAIN':
            params = json.loads(data)
            projectId = params['projectId']
            pyPath = os.path.join(self.work_path, projectId)
            manifest = os.path.join(pyPath, 'manifest.zip')

            self.save_file(tunnel, manifest, filedata)
            self.unzip_file(pyPath, manifest)

            threading.Thread(target=TunnelEventProxy.run_task, args=(self, 'TRAIN', params)).start()

        elif action == b'RUN1':
            params = json.loads(data)
            projectId = params['projectId']
            pyPath = os.path.join(self.work_path, projectId)
            manifest = os.path.join(pyPath, 'manifest.zip')

            self.save_file(tunnel, manifest, filedata)
            self.unzip_file(pyPath, manifest)

            threading.Thread(target=TunnelEventProxy.run_task, args=(self, 'RUN1', params)).start()

        elif action == b'RUN2':
            params = json.loads(data)
            projectId = params['projectId']
            pyPath = os.path.join(self.work_path, projectId)
            manifest = os.path.join(pyPath, 'manifest.zip')

            self.save_file(tunnel, manifest, filedata)
            self.unzip_file(pyPath, manifest)

            threading.Thread(target=TunnelEventProxy.run_task, args=(self, 'RUN2', params)).start()
        
        elif action == b'STOP':
            self.stop_or_not[0] = True
            self.returncode = -2
            print("训练已终止！")
        
        elif action == b'VAL':
            params = json.loads(data)
            projectId = params['projectId']
            pyPath = os.path.join(self.work_path, projectId)
            manifest = os.path.join(pyPath, 'manifest.zip')

            self.save_file(tunnel, manifest, filedata)
            self.unzip_file(pyPath, manifest)

            threading.Thread(target=TunnelEventProxy.run_task, args=(self, 'VAL', params)).start()

    def prepare_face_datasets(self, src_path, target_path):
        for filename in os.listdir(src_path):
            src_subpath = os.path.join(src_path, filename)
            target_subpath = os.path.join(target_path, filename)

            if os.path.isdir(src_subpath):
                if not os.path.exists(target_subpath) and not os.path.isdir(target_subpath):
                    os.mkdir(target_subpath)

                self.prepare_face_datasets(src_subpath, target_subpath)

            else:
                face_detect_and_cut(src_subpath, target_subpath)

    def cleardir(self, path):
        for filename in os.listdir(path):
            subpath = os.path.join(path, filename)

            if os.path.isdir(subpath):
                self.cleardir(subpath)

                os.rmdir(subpath)

            else:
                os.remove(subpath)

    def save_file(self, tunnel, filename, filedata):
        path = os.path.dirname(filename)

        if not os.path.exists(path):
            os.makedirs(path)

        if os.path.exists(filename):
            os.remove(filename)

        tunnel.write_file(filename, filedata)

    def zip_model(self, pyPath, path, filename):
        try:
            py_files = os.listdir(pyPath)
            py_files = [py for py in py_files if py.endswith(".py")]

            pth_files = os.listdir(path)
            pth_files = [pth for pth in pth_files if pth.endswith(".pth")]

            with closing(zipfile.ZipFile(filename, mode="w")) as f:
                for pth in pth_files:
                    f.write(os.path.join(path, pth), pth)

                for py in py_files:
                    f.write(os.path.join(pyPath, py), py)

            return True
        
        except Exception as e:
            traceback.print_exc()

        return False

    def unzip_file(self, path, filename):
        try:
            with closing(zipfile.ZipFile(filename, mode="r")) as f:
                f.extractall(path)

            return True

        except:
            traceback.print_exc()

        return False
    
    def set_datasets_api(self, datasets_api):
        self.datasets_api = datasets_api

    def set_model_api(self, model_api):
        self.model_api = model_api

def typeof(o):
    _type = re.search(r"class \'(\w+)\'", str(type(o)))

    if _type:
        return _type.group(1)

    return None


proxy = TunnelEventProxy()

def TunnelEventHandler(tunnel, action, data, filedata):
    #app.logger.info('TunnelEventHandler', action, data, len(filedata) if filedata else 0)
    print('TunnelEventHandler', action, data, len(filedata) if filedata else 0)

    proxy.run(tunnel, action, data, filedata)

def bbap_tunnel(tunnel, host, port):
    tunnel.run(host, port, TunnelEventHandler)


CORS(app, resourecs=r'/*')   # 实现跨域操作


@app.route('/client/user_login/<username>', methods=['GET'])
def user_login(username):
    #app.logger.info("==> Username: " + username)
    print("==> Username: " + username)

    proxy.username = username

    tunnel.send('LOGIN', username)

    return {
        "result": "success"
    }


def load_json(filename):
    try:
        with closing(open(filename, 'r')) as f:
            return json.load(f)
        
    except Exception as e:
        pass

    return None

def load_conf(filename):
    default_listen = { 'host': '0.0.0.0', 'port': 4998 }
    default_tunnel = { 'host': '127.0.0.1', 'port': 10021 }
    default_server = {
        # 'datasets_api': 'http://localhost:10020/datasets/download/{id}',
        # 'model_api': 'http://localhost:10020/project/model/{id}'
        'datasets_api': 'http://www.aixlab.cn/datasets/download/{id}',
        'model_api': 'http://www.aixlab.cn/project/model/{id}'
    }
    # default_tunnel = { 'host': 'test.aixlab.cn', 'port': 10021 }
    default_tunnel = { 'host': 'www.aixlab.cn', 'port': 10021 }
    conf = load_json(filename)
    if conf:
        if 'listen' not in conf:
            conf['listen'] = default_listen

        if 'tunnel' not in conf:
            conf['tunnel'] = default_tunnel

        if 'server' not in conf:
            conf['server'] = default_server

    else:
        conf = {
            'listen': default_listen,
            'tunnel': default_tunnel,
            'server': default_server
        }

    return conf


if __name__ == "__main__":
    # work_path = os.path.dirname(os.path.realpath(__file__))
    # work_file = os.path.basename(os.path.realpath(__file__))
    # conf_file = os.path.join(work_path, work_file.replace('.py', '.conf'))
    conf_file = './block_client.conf'
    conf = load_conf(conf_file)
    conf_listen = conf['listen']
    conf_tunnel = conf['tunnel']
    conf_server = conf['server']

    threading.Timer(30, TunnelEventProxy.keepalive, args=(proxy, tunnel)).start()

    proxy.set_datasets_api(conf_server['datasets_api'])
    proxy.set_model_api(conf_server['model_api'])

    threading.Thread(target=bbap_tunnel, args=(tunnel, conf_tunnel['host'], conf_tunnel['port'])).start()

    app.run(host=conf_listen['host'], port=conf_listen['port'])

    tunnel.stop()
