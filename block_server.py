from flask import Flask
from flask import request
# from flask_cors import CORS
from main import *
import json
import requests
import time
import threading
import zipfile

from tunnel_server import *
from get_face_imgs import face_detect_and_cut


app = Flask(__name__)

tunnel = TunnelServer()


class TunnelEventProxy():
    def __init__(self):
        self.work_path = os.path.dirname(os.path.realpath(__file__))

    def run(self, tunnel, action, data, filedata):
        if action == b'HEARTBEATS':
            pass

        elif action == b'TRAIN:RESULT':
            params = json.loads(data)
            params['resultPath'] = params['resultPath'].replace('\\', '/')
            resultPath = params['resultPath']

            self.save_file(tunnel, resultPath, filedata)

            response = requests.post("http://127.0.0.1:10020/project/callbackTrain", data=json.dumps(params))

            #app.logger.info(str(response.json()))
            print(str(response.json()))

        elif action == b'RUN1:RESULT':
            params = json.loads(data)
            params['resultPath'] = params['resultPath'].replace('\\', '/')
            resultPath = params['resultPath']

            self.save_file(tunnel, resultPath, filedata)

            response = requests.post("http://127.0.0.1:10020/project/callbackRun", data=json.dumps(params))

            #app.logger.info(str(response.json()))
            print(str(response.json()))

        elif action == b'RUN2:RESULT':
            params = json.loads(data)

            response = requests.post("http://127.0.0.1:10020/project/callbackRun", data=json.dumps(params))

            #app.logger.info(str(response.json()))
            print(str(response.json()))

        elif action == b'VAL:RESULT':
            params = json.loads(data)
            params['resultPath'] = params['resultPath'].replace('\\', '/')
            resultPath = params['resultPath']

            self.save_file(tunnel, resultPath, filedata)

            response = requests.post("http://127.0.0.1:10020/project/callbackVal", data=json.dumps(params))

            #app.logger.info(str(response.json()))
            print(str(response.json()))

    def save_file(self, tunnel, filename, filedata):
        path = os.path.dirname(filename)

        if not os.path.exists(path):
            os.makedirs(path)

        if os.path.exists(filename):
            os.remove(filename)

        tunnel.write_file(filename, filedata)

    @staticmethod
    def ensure_path(filename):
        path = os.path.dirname(filename)

        if not os.path.exists(path):
            os.makedirs(path)

    @staticmethod
    def zip_manifest(path, filename):
        try:
            if os.path.exists(filename):
                os.remove(filename)

            with zipfile.ZipFile(filename, mode="w") as f:
                for fn in os.listdir(path):
                    if fn == 'manifest.zip':
                        continue

                    f.write(os.path.join(path, fn), fn)

        except Exception as e:
            traceback.print_exc()

        finally:
            f.close()

proxy = TunnelEventProxy()

def TunnelEventHandler(tunnel, action, data, filedata):
    #app.logger.info('TunnelEventHandler', action, data, len(filedata) if filedata else 0)
    print('TunnelEventHandler', action, data, len(filedata) if filedata else 0)

    proxy.run(tunnel, action, data, filedata)

def bbap_tunnel(tunnel, host, port):
    tunnel.run(host, port, TunnelEventHandler)


# CORS(app, resourecs=r'/*')   # 实现跨域操作


@app.route('/thumbnail', methods=['POST'])
def thumbnail():
    if request.method == 'POST':
        source = request.form['source']
        target = request.form['target']

        face_detect_and_cut(source, target)

    return {
        "result": "success"
    }


@app.route('/generate_create', methods=['POST'])
def generate_create():
    if request.method == 'POST':
        projectType = request.form['projectType']
        pyPath = request.form['pyPath']

        #app.logger.info("==> projectType: " + projectType + ", pyPath: " + pyPath)
        print("==> projectType: " + projectType + ", pyPath: " + pyPath)

        TunnelEventProxy.ensure_path(pyPath)

        generate_o(pyPath)

    return {
        "result": "success"
    }


@app.route('/generate_save', methods=['POST'])
def generate_save():
    if request.method == 'POST':
        projectType = request.form['projectType']
        pyPath = request.form['pyPath']
        json_argument = request.form['json_argument']
        new_dict = json.loads(json_argument)

        #with open("D:/azhi/bbap/Block/record.json", "w") as f:
        #    json.dump(new_dict, f)

        #app.logger.info("==> projectType: " + projectType + ", pyPath: " + pyPath + ", " + json_argument)
        print("==> projectType: " + projectType + ", pyPath: " + pyPath + ", " + json_argument)

        TunnelEventProxy.ensure_path(pyPath)

        generate(pyPath, projectType, json_argument)

    return {
        "result": "success"
    }


@app.route('/train_model', methods=['POST'])
def train_model():
    if request.method == 'POST':
        username = request.form['username']
        projectType = request.form['projectType']
        projectId = request.form['projectId']
        pyPath = request.form['pyPath']
        folderPath = request.form['folderPath']
        datasetsId = request.form['datasetsId']
        manifest = os.path.join(pyPath, 'manifest.zip')

        #app.logger.info("==> username: " + username + ", projectId: " + projectId + ", pyPath: " + pyPath + ", path: " + folderPath)
        print("==> username: " + username + ", projectType: " + projectType + ", projectId: " + projectId + ", pyPath: " + pyPath + ", path: " + folderPath + ", datasetsId: " + datasetsId)

        TunnelEventProxy.zip_manifest(pyPath, manifest)

        tunnel.send_file(username, 'TRAIN', json.dumps({
            'projectType': projectType,
            'projectId': projectId,
            'pyPath': pyPath,
            'folderPath': folderPath,
            'datasetsId': datasetsId
        }), manifest)

    return {
        "result": "success"
    }


@app.route('/run_project', methods=['POST'])
def run_project():
    if request.method == 'POST':
        username = request.form['username']
        projectType = request.form['projectType']
        projectId = request.form['projectId']
        pyPath = request.form['pyPath']
        manifest = os.path.join(pyPath, 'manifest.zip')

        #app.logger.info("==> username: " + username + ", projectId: " + projectId + ", projectType: " + projectType + ", pyPath: " + pyPath)
        print("==> username: " + username + ", projectType: " + projectType + ", projectId: " + projectId + ", pyPath: " + pyPath)

        TunnelEventProxy.zip_manifest(pyPath, manifest)

        if (int(projectType)) == 1:
            filePath = request.form['filePath']
            datasetsId = request.form['datasetsId']

            print("==> filePath: " + filePath + ", datasetsId: " + datasetsId)

            tunnel.send_file(username, 'RUN1', json.dumps({
                'projectType': projectType,
                'projectId': projectId,
                'pyPath': pyPath,
                'filePath': filePath,
                'datasetsId': datasetsId
            }), manifest)

            return {
                "result": "success",
                "msg": "运行成功"
            }

        elif (int(projectType)) == 2:
            tunnel.send_file(username, 'RUN2', json.dumps({
                'projectType': projectType,
                'projectId': projectId,
                'pyPath': pyPath
            }), manifest)

            return {
                "result": "success",
                "msg": "运行成功"
            }

    return {
        "result": "success",
        "msg": "Type Error"
    }

@app.route('/stop_train', methods=['POST'])
def stop_project():
    if request.method == 'POST':
        username = request.form['username']
        projectId = request.form['projectId']
        pyPath = request.form['pyPath']
        manifest = os.path.join(pyPath, 'manifest.zip')
        
        tunnel.send_file(username, 'STOP', json.dumps({
            'projectId': projectId,
            'pyPath': pyPath,
        }), manifest)

        return {
            "result": "success",
            "msg": "f{projectId}已停止"
        }

@app.route('/val_project', methods=['POST'])
def val_project():
    if request.method == 'POST':
        username = request.form['username']
        projectType = request.form['projectType']
        projectId = request.form['projectId']
        pyPath = request.form['pyPath']
        manifest = os.path.join(pyPath, 'manifest.zip')

        print("==> username: " + username + ", projectType: " + projectType + ", projectId: " + projectId + ", pyPath: " + pyPath)

        TunnelEventProxy.zip_manifest(pyPath, manifest)

        filePath = request.form['filePath']
        datasetsId = request.form['datasetsId']

        print("==> filePath: " + filePath + ", datasetsId: " + datasetsId)

        tunnel.send_file(username, 'VAL', json.dumps({
            'projectType': projectType,
            'projectId': projectId,
            'pyPath': pyPath,
            'filePath': filePath,
            'datasetsId': datasetsId
        }), manifest)

        return {
            "result": "success",
            "msg": "验证成功"
        }

    return {
        "result": "success",
        "msg": "Type Error"
    }

if __name__ == "__main__":
    threading.Thread(target=bbap_tunnel, args=(tunnel, '0.0.0.0', 10021)).start()

    app.run(host='0.0.0.0', port=4999)

    tunnel.stop()
