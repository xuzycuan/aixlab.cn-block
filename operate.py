# -*- coding: UTF-8 -*-
import os
import json
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import random
import math
import numpy as np
import scipy.io.wavfile as wav
from scipy.fftpack import dct
from PIL import Image
from matplotlib import pyplot as plt
plt.switch_backend('agg')
import torch
from torch import nn
from torch import optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

#设置变量
IMG_OR_AUD = 'Img'
CLASS_NUM = 2
AUD_TO_IMG = False
GEST_REC = False
FACE_REC = True
BATCHSIZE = 5
LAYER_NUM = 1
LAYER_TYPE = [None, None, None, None, None]
LAYER_ARGS = [{}, {}, {}, {}, {}]
LAYER_TYPE[0] = 'Fc'
LAYER_ARGS[0]['ACTIVATION_FUN'] = 'IdentityFunction'
LAYER_ARGS[0]['IN_DIMENSION'] = 0
LAYER_ARGS[0]['OUT_DIMENSION'] = 0
LAYER_ARGS[0]['IN_CHANNEL'] = 0
LAYER_ARGS[0]['OUT_CHANNEL'] = 0
LAYER_ARGS[0]['KERNEL_SIZE'] = 0
LAYER_ARGS[0]['STRIDE'] = 0
LAYER_ARGS[0]['PADDING'] = 0
LAYER_ARGS[0]['POOL'] = False
LAYER_TYPE[1] = None
LAYER_ARGS[1]['ACTIVATION_FUN'] = 'ReLU'
LAYER_ARGS[1]['IN_DIMENSION'] = 0
LAYER_ARGS[1]['OUT_DIMENSION'] = 0
LAYER_ARGS[1]['IN_CHANNEL'] = 0
LAYER_ARGS[1]['OUT_CHANNEL'] = 0
LAYER_ARGS[1]['KERNEL_SIZE'] = 0
LAYER_ARGS[1]['STRIDE'] = 0
LAYER_ARGS[1]['PADDING'] = 0
LAYER_ARGS[1]['POOL'] = False
LAYER_TYPE[2] = None
LAYER_ARGS[2]['ACTIVATION_FUN'] = 'ReLU'
LAYER_ARGS[2]['IN_DIMENSION'] = 0
LAYER_ARGS[2]['OUT_DIMENSION'] = 0
LAYER_ARGS[2]['IN_CHANNEL'] = 0
LAYER_ARGS[2]['OUT_CHANNEL'] = 0
LAYER_ARGS[2]['KERNEL_SIZE'] = 0
LAYER_ARGS[2]['STRIDE'] = 0
LAYER_ARGS[2]['PADDING'] = 0
LAYER_ARGS[2]['POOL'] = False
LAYER_TYPE[3] = None
LAYER_ARGS[3]['ACTIVATION_FUN'] = 'ReLU'
LAYER_ARGS[3]['IN_DIMENSION'] = 0
LAYER_ARGS[3]['OUT_DIMENSION'] = 0
LAYER_ARGS[3]['IN_CHANNEL'] = 0
LAYER_ARGS[3]['OUT_CHANNEL'] = 0
LAYER_ARGS[3]['KERNEL_SIZE'] = 0
LAYER_ARGS[3]['STRIDE'] = 0
LAYER_ARGS[3]['PADDING'] = 0
LAYER_ARGS[3]['POOL'] = False
LAYER_TYPE[4] = None
LAYER_ARGS[4]['ACTIVATION_FUN'] = 'ReLU'
LAYER_ARGS[4]['IN_DIMENSION'] = 0
LAYER_ARGS[4]['OUT_DIMENSION'] = 0
LAYER_ARGS[4]['IN_CHANNEL'] = 0
LAYER_ARGS[4]['OUT_CHANNEL'] = 0
LAYER_ARGS[4]['KERNEL_SIZE'] = 0
LAYER_ARGS[4]['STRIDE'] = 0
LAYER_ARGS[4]['PADDING'] = 0
LAYER_ARGS[4]['POOL'] = False
LEARNING_RATE = 0
OPTIMIZER = 'SGD'
EPOCHS = 0
LOSS_FUN = 'CE'
RATIO_V = 0

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.class_num = CLASS_NUM
        self.layer1 = make_layer(LAYER_TYPE[0], LAYER_ARGS[0])
        self.layer2 = make_layer(LAYER_TYPE[1], LAYER_ARGS[1])
        self.layer3 = make_layer(LAYER_TYPE[2], LAYER_ARGS[2])
        self.layer4 = make_layer(LAYER_TYPE[3], LAYER_ARGS[3])
        self.layer5 = make_layer(LAYER_TYPE[4], LAYER_ARGS[4])

    def forward(self, x):
        if self.layer1:
            if LAYER_TYPE[0] == 'Fc':
                x = x.view(x.size(0), -1)
            x = self.layer1(x)
        if self.layer2:
            if LAYER_TYPE[1] == 'Fc':
                x = x.view(x.size(0), -1)
            x = self.layer2(x)
        if self.layer3:
            if LAYER_TYPE[2] == 'Fc':
                x = x.view(x.size(0), -1)
            x = self.layer3(x)
        if self.layer4:
            if LAYER_TYPE[3] == 'Fc':
                x = x.view(x.size(0), -1)
            x = self.layer4(x)
        if self.layer5:
            if LAYER_TYPE[4] == 'Fc':
                x = x.view(x.size(0), -1)
            x = self.layer5(x)
        out = x.view(x.size(0), -1)
        return out

# set var
dic = {}

def set_var(path_py):
    path = os.path.abspath(os.path.join(path_py, os.pardir))
    with open(path + '\\generated.json', 'r') as file:
        parameters = json.load(file)
        for item in parameters:
            param = item['param']
            value = item['value']
            if value == "false" or value == "False":
                value = False
            dic[param] = value

    for key, value in dic.items():
        globals()[key] = value
        exec(f"{key} = {repr(value)}")

    net = Net()
    epochs = EPOCHS
    lr = LEARNING_RATE
    if OPTIMIZER == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=lr)
    if OPTIMIZER == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=lr)
    if LOSS_FUN == 'CE':
        criterion = nn.CrossEntropyLoss()
    if LOSS_FUN == 'MSE':
        criterion = nn.MSELoss()

    return net, epochs, optimizer, criterion
    
# ReadData
def read_data(is_train, data_path):
    data_output = []
    if is_train:
        data_output = []
        classes = []
        ct_label = 0
        for class_name in os.listdir(data_path):
            classes.append(class_name)
            for root, dirs, files in os.walk(os.path.join(data_path, class_name)):
                for file in files:
                    data_file = None
                    data_filename = os.path.join(data_path, class_name, file)

                    try:
                        print("Read: %s" %data_filename)

                        if IMG_OR_AUD == 'Img':
                            data_file = Image.open(data_filename).convert('RGB')
                            data_file = np.array(data_file)

                        if IMG_OR_AUD == 'Aud':
                            data_file = wav.read(data_filename)

                    except Exception as e:
                        print(" err:", e)
                        continue

                    data_output.append((data_file, ct_label))

            ct_label += 1
    else:
        classes = None
        data_file = None
        if IMG_OR_AUD == 'Img':
            data_file = Image.open(data_path)
            data_file = np.array(data_file)

        if IMG_OR_AUD == 'Aud':
            data_file = wav.read(data_path)
        data_output.append((data_file, 0))
    return data_output, classes


""" GENERATE ReadData END """

""" GENERATE AudToImg START """


# AudToImg
class Signal(object):
    def __init__(self, y, sampling_freq, t=None):
        self.y = y
        self.sampling_freq = sampling_freq
        if t is None:
            self.t = np.arange(len(y)) / sampling_freq
        else:
            self.t = t

    def get_short_term_energy(self, window_size):
        window = np.hamming(window_size)
        i, j = 0, window_size
        step = window_size // 2
        energe = []
        t = []
        while j < len(self.y):
            frame = self.y[i: j]
            frame_add_win = frame * window
            energe.append(np.sum(frame_add_win * frame_add_win))
            t.append((self.t[i] + self.t[j]) / 2)
            i += step
            j += step
        return Signal(energe, sampling_freq=1 / (t[1] - t[0]), t=t)

    def get_short_term_ZCR(self, window_size):
        window = np.hamming(window_size)
        i, j = 0, window_size
        step = window_size // 2
        sign = lambda num: 1 if num > 0 else -1 if num < 0 else 0
        zcr = []
        t = []
        while j < len(self.y):
            frame = self.y[i: j]
            f_w = frame * window
            s_f_w = np.array([sign(num) for num in f_w])
            zcr.append(np.sum(np.abs(s_f_w[1:] - s_f_w[:window_size - 1])) / 2)
            t.append((self.t[i] + self.t[j]) / 2)
            i += step
            j += step
        return Signal(zcr, sampling_freq=1 / (t[1] - t[0]), t=t)

    def endpoint_detection(self, draw=False):
        frame_len = 0.01
        window_size = math.ceil(frame_len * self.sampling_freq)
        energe = self.get_short_term_energy(window_size)
        zcr = self.get_short_term_ZCR(window_size)
        EMax, EMin, C = 2, 1, 40

        # EMax, EMin, C = 1, 1, 100

        def get_endpoint_index(energe, zcr):
            zcr_len = len(zcr)
            point_h = zcr_len - 1
            for i in range(zcr_len):
                if energe[i] > EMax:
                    point_h = i
                    break
            point = point_h
            for i in range(point_h, -1, -1):
                if energe[i] < EMin:
                    point = i
                    break
            for i in range(point, -1, -1):
                if zcr[i] < C:
                    return i
            return point

        begin = energe.t[get_endpoint_index(energe.y, zcr.y)]
        end = energe.t[::-1][get_endpoint_index(energe.y[::-1], zcr.y[::-1])]

        def get_endpoint_time(time):
            for i in range(len(self.t)):
                if self.t[i] > time:
                    return i
            return 0

        begin = get_endpoint_time(begin)
        end = get_endpoint_time(end)
        return begin, end

    def pre_emphasis(self, alpha=0.97):
        y = self.y
        self.y = np.append(y[0], y[1:] - alpha * y[:-1])

    def get_mfcc(self, draw=False, num_mel=128, num_cep=12, init_sed=20):
        sampling_freq = self.sampling_freq
        # 端点检测
        begin, end = self.endpoint_detection(draw=draw)
        # 预加重
        self.pre_emphasis()
        self.y = np.append(self.y[begin:end + 1], np.zeros(sampling_freq * init_sed - (end - begin)))
        y = self.y
        # 分帧
        sig_len = len(y)
        frame_size, frame_stride = 0.03, 0.01
        frame_len, frame_step = round(frame_size * sampling_freq), round(frame_stride * sampling_freq)
        num_frames = math.ceil((sig_len - frame_len) / frame_step)
        if math.ceil((end - begin - frame_len) / frame_step) <= 0:
            return False, False
        pad_sig_len = num_frames * frame_step + frame_len
        pad_sig = np.append(y, np.zeros((pad_sig_len - sig_len)))
        num_frames += 1
        frames = np.hstack(
            [pad_sig[i * frame_step: i * frame_step + frame_len].reshape(-1, 1) for i in range(0, num_frames)])
        # 加窗
        window = np.hamming(frame_len)
        frames = frames * window.reshape(-1, 1)
        # fft + 能量谱
        # 区别？？
        frames = np.abs(np.fft.rfft(frames)) ** 2
        # mel滤波
        mel_freq = 2595 * np.log10(1 + sampling_freq / 2 / 700)
        mel_points = np.linspace(0, mel_freq, num_mel + 2)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        freq_index = np.floor((frame_len + 1) / sampling_freq * hz_points).astype(int)
        mel_filters = np.zeros((num_mel, frame_len))
        for i in range(1, num_mel + 1):
            l = freq_index[i - 1]
            m = freq_index[i]
            r = freq_index[i + 1]
            for k in range(l, m):
                mel_filters[i - 1][k] = (k - l) / (m - l)
            for k in range(m, r):
                mel_filters[i - 1][k] = (r - k) / (r - m)
        mel_feat = mel_filters.dot(frames)
        # mel_feat = 20 * np.log10(mel_feat)
        mfcc = dct(mel_feat, type=2, axis=1, norm='ortho')[1: num_cep + 1, :]
        mel_filters -= np.mean(mel_filters, 1, keepdims=True)
        mfcc -= np.mean(mfcc, 1, keepdims=True)
        return mel_feat, mfcc


def normalize(sig, amp=1.0):
    high, low = abs(max(sig)), abs((min(sig)))
    return amp * sig / max(high, low)


def delete_noisy(sig, num1=3000, num2=5000):
    a = np.append(np.zeros((num1)), sig[num1:len(sig) - num2])
    return np.append(a, np.zeros((num2)))


def audio_to_img(raw_data):
    new_data = []
    for wav_file, label in raw_data:
        sampling_freq, sig = wav_file
        sig = delete_noisy(sig)
        mfcc_feat, _ = Signal(normalize(sig), sampling_freq=sampling_freq).get_mfcc()
        new_data.append((mfcc_feat, label))
    return new_data


""" GENERATE AudToImg END """

""" GENERATE ImgPreProcess START"""
# ImgPreProcess
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToPILImage(),
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
])
# transform = transforms.Compose([
#         # transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.Resize((32, 32)),
#         transforms.ToTensor(),
# ])

class MyDataset(Dataset):
    def __init__(self, data_cur, transform=None):
        self.data_cur = data_cur
        self.length = len(data_cur)
        self.transform = transform

    def __getitem__(self, idx):
        img = self.transform(self.data_cur[idx][0])
        return img, self.data_cur[idx][1]

    def __len__(self):
        return self.length


def data_preprocess_run(data_cur):
    data_cur2 = []
    for img, label in data_cur:
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2).repeat(3, 2)
        img = img.astype(np.float32)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        data_cur2.append((img, label))
    dataset = MyDataset(data_cur2, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCHSIZE, shuffle=True, num_workers=0)

    return dataloader

def data_preprocess(data_cur):
    data_cur2 = []
    for img, label in data_cur:
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2).repeat(3, 2)
        img = img.astype(np.float32)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        data_cur2.append((img, label))
    
    # 手动划分训练集和测试集
    random.shuffle(data_cur2)
    split_idx = int((1-RATIO_V) * len(data_cur2))
    train_data = data_cur2[:split_idx]
    test_data = data_cur2[split_idx:]
    
    train_dataset = MyDataset(train_data, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True, num_workers=0)
    
    test_dataset = MyDataset(test_data, transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCHSIZE, shuffle=False, num_workers=0)

    return train_dataloader, test_dataloader

""" GENERATE ImgPreProcess END """

""" GENERATE NeuronNet START """


# NeuronNet
def make_layer(layer_type, layer_args):
    activation_fun = None
    if layer_args['ACTIVATION_FUN'] == 'ReLU':
        activation_fun = nn.ReLU()
    if layer_args['ACTIVATION_FUN'] == 'Sigmoid':
        activation_fun = nn.Sigmoid()

    if layer_type == 'Conv':
        if activation_fun:
            if layer_args['POOL']:
                return nn.Sequential(
                    nn.Conv2d(in_channels=layer_args['IN_CHANNEL'],
                            out_channels=layer_args['OUT_CHANNEL'],
                            kernel_size=layer_args['KERNEL_SIZE'],
                            stride=layer_args['STRIDE'],
                            padding=layer_args['PADDING']),
                    activation_fun, nn.MaxPool2d(kernel_size=2,stride = 2)
                )
            else:
                return nn.Sequential(
                    nn.Conv2d(in_channels=layer_args['IN_CHANNEL'],
                            out_channels=layer_args['OUT_CHANNEL'],
                            kernel_size=layer_args['KERNEL_SIZE'],
                            stride=layer_args['STRIDE'],
                            padding=layer_args['PADDING']),
                    activation_fun
                )
        else:
            if layer_args['POOL']:
                return nn.Sequential(
                    nn.Conv2d(in_channels=layer_args['IN_CHANNEL'],
                            out_channels=layer_args['OUT_CHANNEL'],
                            kernel_size=layer_args['KERNEL_SIZE'],
                            stride=layer_args['STRIDE'],
                            padding=layer_args['PADDING']),
                    nn.MaxPool2d(kernel_size=2)
                )
            else:
                return nn.Sequential(
                    nn.Conv2d(in_channels=layer_args['IN_CHANNEL'],
                            out_channels=layer_args['OUT_CHANNEL'],
                            kernel_size=layer_args['KERNEL_SIZE'],
                            stride=layer_args['STRIDE'],
                            padding=layer_args['PADDING']),
                )
    if layer_type == 'Fc':
        if activation_fun:
            return nn.Sequential(
                nn.Linear(layer_args['IN_DIMENSION'],
                        layer_args['OUT_DIMENSION']),
                activation_fun
            )
        else:
            return nn.Sequential(
                nn.Linear(layer_args['IN_DIMENSION'],
                        layer_args['OUT_DIMENSION']),
            )
    return None


""" GENERATE Train START """


# Train
def train(data_path, result_path, stop_or_not):
    net, epochs, optimizer, criterion = set_var(result_path)
    print('Train')
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    if not os.path.isdir("./Result"):
        os.mkdir("./Result")

    data_cur, classes = read_data(True, data_path)
    if AUD_TO_IMG:
        data_cur = audio_to_img(data_cur)
    train_loader, test_loader = data_preprocess(data_cur)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    criterion.to(device)

    net.train()
    loss_list = []
    acc_list = []
    batch_id = 0
    for epoch in range(epochs):

        train_loss = 0
        correct = 0
        total = 0
        for batch_id, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets.long())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if batch_id % 10 == 0:
                print("epoch:", epoch + 1, batch_id, 'Loss: %.3f | Acc: %.3f (%d/%d)' % (
                train_loss / (batch_id + 1), 100. * correct / total, correct, total))
        loss_list.append(train_loss / (batch_id + 1))
        acc_list.append(100. * correct / total)
        net.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for val_inputs, val_targets in test_loader:
                val_inputs = val_inputs.to(device)
                val_targets = val_targets.to(device)
                val_outputs = net(val_inputs)
                _, val_predicted = val_outputs.max(1)
                val_total += val_targets.size(0)
                val_correct += val_predicted.eq(val_targets).sum().item()
        if val_total == 0:
            val_total = 1
        val_accuracy = 100. * val_correct / val_total
        print('Validation Acc: %.3f (%d/%d)' % (val_accuracy, val_correct, val_total))

    x = np.arange(epochs)
    y1 = np.array(loss_list)
    y2 = np.array(acc_list)
    plt.clf()
    plt.plot(x, y1, x, y2)
    plt.savefig(os.path.join(result_path, 'result.jpg'))

    torch.save(net.state_dict(), os.path.join(result_path, 'params.pth'))
    torch.save(classes, os.path.join(result_path, 'classes.pth'))
    with open('Result/train_result.json', 'w') as file:
        json.dump(round(val_accuracy, 2), file)
    return round(val_accuracy, 2)

""" GENERATE Run END """

""" GENERATE Val START """


# Val
def val(data_path, result_path):
    print('Val')
    net, epochs, optimizer, criterion = set_var(result_path)
    print(os.path.join(result_path, 'params.pth'))
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    if not os.path.isdir("./Result"):
        os.mkdir("./Result")

    data_cur, classes = read_data(True, data_path)
    if AUD_TO_IMG:
        data_cur = audio_to_img(data_cur)
    data_loader = data_preprocess_run(data_cur)

    with open(os.path.join(result_path, 'val_result.txt'), mode='w', encoding='utf-8') as f:
        
        with torch.no_grad():
            net.load_state_dict(torch.load(os.path.join(result_path, 'params.pth')))
            net.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for val_inputs, val_targets in data_loader:
                val_outputs = net(val_inputs)
                _, val_predicted = val_outputs.max(1)
                val_total += val_targets.size(0)
                val_correct += val_predicted.eq(val_targets).sum().item()
        val_accuracy = 100. * val_correct / val_total
        print('Validation Acc: %.3f (%d/%d)' % (val_accuracy, val_correct, val_total))
        print(val_accuracy ,file= f)
    with open('Result/val_result.json', 'w') as file:
        json.dump(round(val_accuracy,2), file)
    return round(val_accuracy,2)

# is_running = True

# def stop_training():
#     global is_running
#     is_running = False
""" GENERATE Val END """

if __name__ == '__main__':
    # # 读参数文件
    # with open('args.txt', 'r') as f:
    #     lines = f.readlines()
    #     a = ''
    #     for line in lines:
    #         a += line.strip()
    # a = a.split()
    # a = ''.join(a)

    # path_py = 'C:\\Users\\HUAWEI\\Desktop\\block\\1725427424059068416\\1'
    # path_folder = 'C:\\Users\\HUAWEI\\Desktop\\block\\1725427424059068416\\train_datas'
    # train(path_folder,path_py,1)
    if len(sys.argv) > 1:
        if sys.argv[1] == 'train':
            train(sys.argv[2], sys.argv[3])
        elif sys.argv[1] == 'run':
            run(sys.argv[2], sys.argv[3])
        elif sys.argv[1] == 'val':
            val(sys.argv[2], sys.argv[3])
        # elif sys.argv[1] == 'stop':
        #     stop_training()
        else:
            print('Invalid function name')
    else:
        print('Function name not provided')