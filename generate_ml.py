# -*- coding: UTF-8 -*-
import os


# 神经网络参数
class ArgumentsMLNN:
    def __init__(self):
        self.LAYER_TYPE = None
        self.LAYER_ARGS = {'ACTIVATION_FUN': 'ReLU',
                           'IN_DIMENSION': 0,
                           'OUT_DIMENSION': 0,
                           'IN_CHANNEL': 3,
                           'OUT_CHANNEL': 16,
                           'KERNEL_SIZE': 3,
                           'STRIDE': 1,
                           'PADDING': 1,
                           'POOL': False}

# ML类型项目参数
class ArgumentsMl:
    def __init__(self):
        self.IMG_OR_AUD = 'Img'
        self.CLASS_NUM = 0
        self.AUD_TO_IMG = False
        self.GEST_REC = False
        self.FACE_REC = False
        self.BATCHSIZE = 5
        self.RATIO_V = 0.2
        self.MAX_LAYER_NUM = 5
        self.LAYER_NUM = 5
        self.MODEL_ARGS = []
        self.LAYER1 = ArgumentsMLNN()
        self.MODEL_ARGS.append(self.LAYER1)
        self.LAYER2 = ArgumentsMLNN()
        self.MODEL_ARGS.append(self.LAYER2)
        self.LAYER3 = ArgumentsMLNN()
        self.MODEL_ARGS.append(self.LAYER3)
        self.LAYER4 = ArgumentsMLNN()
        self.MODEL_ARGS.append(self.LAYER4)
        self.LAYER5 = ArgumentsMLNN()
        self.MODEL_ARGS.append(self.LAYER5)
        self.LEARNING_RATE = 0.1
        self.OPTIMIZER = 'SGD'
        self.EPOCHS = 20
        self.LOSS_FUN = 'CE'


def generate_ml(target_path, args):
    print(args.get('data'))
    a = args.get('data')['algorithmList']
    args = ArgumentsMl()

    if os.path.exists(target_path+"generated.py"):
        os.remove(target_path+"generated.py")

    with open(os.path.join('Templates_ml', 'template_ml.py'), 'r', encoding='utf-8', errors='ignore') as f_template:
        print(f_template)
        lines = f_template.readlines()
        with open(target_path, mode='w', encoding='utf-8') as f_target:

            #
            # 写文件头 done
            flag = False
            for line in lines:
                if 'GENERATE Import START' in line:
                    flag = True
                if flag:
                    print(line, end='', file=f_target)
                if 'GENERATE Import END' in line:
                    flag = False
            print(file=f_target)

            #
            # 写参数 done
            for item in a:
                for k in item:
                    if k == 'id' and item[k] == 1003:
                        for item2 in item['pList']:
                            if item2['param'] == 'CLASS_NUM':
                                args.CLASS_NUM = item2['value']
                            if item2['param'] == 'IMG_OR_AUD':
                                args.IMG_OR_AUD = item2['value']
                    if k == 'id' and item[k] == 1004:
                        args.AUD_TO_IMG = True
                    if k == 'id' and item[k] == 1005:
                        args.GEST_REC = True
                    if k == 'id' and item[k] == 1006:
                        args.FACE_REC = True
                    if k == 'id' and item[k] == 1007:
                        for item2 in item['pList']:
                            if item2['param'] == 'BATCHSIZE':
                                args.BATCHSIZE = item2['value']
                            if item2['param'] == 'RATIO_V':
                                args.RATIO_V = item2['value']
                    if k == 'id' and item[k] == 1008:
                        args.LAYER_NUM = len(item['blockList'])
                        ct = 0
                        for item2 in item['blockList']:
                            if item2['blockId'] == 1010:
                                args.MODEL_ARGS[ct].LAYER_TYPE = 'Conv'
                            if item2['blockId'] == 1009:
                                args.MODEL_ARGS[ct].LAYER_TYPE = 'Fc'
                            for item3 in item2['pList']:
                                if item3['param'] == 'ACTIVATION_FUN':
                                    args.MODEL_ARGS[ct].LAYER_ARGS['ACTIVATION_FUN'] = item3['value']
                                if item3['param'] == 'IN_DIMENSION':
                                    args.MODEL_ARGS[ct].LAYER_ARGS['IN_DIMENSION'] = item3['value']
                                if item3['param'] == 'OUT_DIMENSION':
                                    args.MODEL_ARGS[ct].LAYER_ARGS['OUT_DIMENSION'] = item3['value']
                                if item3['param'] == 'IN_CHANNEL':
                                    args.MODEL_ARGS[ct].LAYER_ARGS['IN_CHANNEL'] = item3['value']
                                if item3['param'] == 'OUT_CHANNEL':
                                    args.MODEL_ARGS[ct].LAYER_ARGS['OUT_CHANNEL'] = item3['value']
                                if item3['param'] == 'KERNEL_SIZE':
                                    args.MODEL_ARGS[ct].LAYER_ARGS['KERNEL_SIZE'] = item3['value']
                                if item3['param'] == 'STRIDE':
                                    args.MODEL_ARGS[ct].LAYER_ARGS['STRIDE'] = item3['value']
                                if item3['param'] == 'PADDING':
                                    args.MODEL_ARGS[ct].LAYER_ARGS['PADDING'] = item3['value']
                                if item3['param'] == 'POOL':
                                    args.MODEL_ARGS[ct].LAYER_ARGS['POOL'] = item3['value']

                            ct += 1

                    if k == 'id' and item[k] == 1011:
                        for item2 in item['pList']:
                            if item2['param'] == 'LEARNING_RATE':
                                args.LEARNING_RATE = item2['value']
                            if item2['param'] == 'OPTIMIZER':
                                args.OPTIMIZER = item2['value']
                            if item2['param'] == 'EPOCHS':
                                args.EPOCHS = item2['value']
                            if item2['param'] == 'LOSS_FUN':
                                args.LOSS_FUN = item2['value']

            print('""" GENERATE Arguments START """', file=f_target)
            print('# Block Arguments', file=f_target)
            print('IMG_OR_AUD = ' + "'" + str(args.IMG_OR_AUD) + "'", file=f_target)
            print('CLASS_NUM = ' + str(args.CLASS_NUM), file=f_target)
            print('AUD_TO_IMG = ' + str(args.AUD_TO_IMG), file=f_target)
            print('GEST_REC = ' + str(args.GEST_REC), file=f_target)
            print('FACE_REC = ' + str(args.FACE_REC), file=f_target)
            print('BATCHSIZE = ' + str(args.BATCHSIZE), file=f_target)
            print('LAYER_NUM = ' + str(args.LAYER_NUM), file=f_target)
            print('LAYER_TYPE = [None, None, None, None, None]', file=f_target)
            print('LAYER_ARGS = [{}, {}, {}, {}, {}]', file=f_target)

            for i in range(args.MAX_LAYER_NUM):
                if args.MODEL_ARGS[i].LAYER_TYPE:
                    print('LAYER_TYPE[' + str(i) + "] = '" + str(args.MODEL_ARGS[i].LAYER_TYPE) + "'", file=f_target)
                else:
                    print('LAYER_TYPE[' + str(i) + "] = None", file=f_target)

                for k in args.MODEL_ARGS[i].LAYER_ARGS:
                    if args.MODEL_ARGS[i].LAYER_ARGS[k] == 'true':
                        args.MODEL_ARGS[i].LAYER_ARGS[k] = True
                    if args.MODEL_ARGS[i].LAYER_ARGS[k] == 'false':
                        args.MODEL_ARGS[i].LAYER_ARGS[k] = False
                    if k in ['ACTIVATION_FUN']:
                        args.MODEL_ARGS[i].LAYER_ARGS[k] = "'" + args.MODEL_ARGS[i].LAYER_ARGS[k] + "'"
                    print('LAYER_ARGS[' + str(i) + "]['" + k + "'] = " + str(args.MODEL_ARGS[i].LAYER_ARGS[k]), file=f_target)

            print('LEARNING_RATE = ' + str(args.LEARNING_RATE), file=f_target)
            print('OPTIMIZER = ' + "'" + str(args.OPTIMIZER) + "'", file=f_target)
            print('EPOCHS = ' + str(args.EPOCHS), file=f_target)
            print('LOSS_FUN = ' + "'" + str(args.LOSS_FUN) + "'", file=f_target)
            print('RATIO_V = ' + str(args.RATIO_V), file=f_target)
            print('""" GENERATE Arguments END """\n', file=f_target)

            #
            # 写各个模块 done
            for item in a:
                for k in item:
                    if k == 'id' and item[k] == 1003:
                        flag = False
                        for line in lines:
                            if 'GENERATE ReadData START' in line:
                                flag = True
                            if flag:
                                print(line, end='', file=f_target)
                            if 'GENERATE ReadData END' in line:
                                flag = False
                        print(file=f_target)

                    if k == 'id' and item[k] == 1004:
                        flag = False
                        for line in lines:
                            if 'GENERATE AudToImg START' in line:
                                flag = True
                            if flag:
                                print(line, end='', file=f_target)
                            if 'GENERATE AudToImg END' in line:
                                flag = False
                        print(file=f_target)

                    if k == 'id' and item[k] == 1007:
                        flag = False
                        for line in lines:
                            if 'GENERATE ImgPreProcess START' in line:
                                flag = True
                            if flag:
                                print(line, end='', file=f_target)
                            if 'GENERATE ImgPreProcess END' in line:
                                flag = False
                        print(file=f_target)

                    if k == 'id' and item[k] == 1008:
                        flag = False
                        for line in lines:
                            if 'GENERATE NeuronNet START' in line:
                                flag = True
                            if flag:
                                print(line, end='', file=f_target)
                            if 'GENERATE NeuronNet END' in line:
                                flag = False
                        print(file=f_target)

                    if k == 'id' and item[k] == 1011:
                        flag = False
                        for line in lines:
                            if 'GENERATE LearningAlg START' in line:
                                flag = True
                            if flag:
                                print(line, end='', file=f_target)
                            if 'GENERATE LearningAlg END' in line:
                                flag = False
                        print(file=f_target)

            # 写训练与运行接口，并写入结尾测试部分
            flag = False
            for line in lines:
                if 'GENERATE Train START' in line:
                    flag = True
                if flag:
                    print(line, end='', file=f_target)
                if 'GENERATE Train END' in line:
                    flag = False
            print(file=f_target)

            flag = False
            for line in lines:
                if 'GENERATE Run START' in line:
                    flag = True
                if flag:
                    print(line, end='', file=f_target)
                if 'GENERATE Run END' in line:
                    flag = False
            print(file=f_target)

            # 写验证接口，并写入结尾测试部分
            flag = False
            for line in lines:
                if 'GENERATE Val START' in line:
                    flag = True
                if flag:
                    print(line, end='', file=f_target)
                if 'GENERATE Val END' in line:
                    flag = False
            print(file=f_target)

            print("if __name__ == '__main__':", file=f_target)
            print("    train_path = 'Figuredata'", file=f_target)
            print("    result_path = 'Result'", file=f_target)
            print("    # run_path = 'Testset/back.jpg'", file=f_target)
            print("    train(train_path, result_path)", file=f_target)
            print("    # run(run_path)", file=f_target)


if __name__ == '__main__':
    generate_ml('', None)
