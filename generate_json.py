# -*- coding: UTF-8 -*-
import os
import json

def extract_params(data):
    params = []
    layer_count = 0
    print (data)
    for item in data:
        if item.get("id") == 1004:
            params.append({"param": "AUD_TO_IMG", "value": True})
        if "pList" in item:
            pList = item["pList"]
            for param_item in pList:
                param_key = param_item["param"]
                param_value = param_item["value"]
                if param_value is not None:
                    if param_key in ["CLASS_NUM", "BATCHSIZE",  "EPOCHS"]:
                        param_value = int(param_value)
                    elif param_key in ["RATIO_V","LEARNING_RATE",]:
                        param_value = float(param_value)
                param = {"param": param_key, "value": param_value}
                params.append(param)
        elif "blockList" in item:
            blockList = item["blockList"]
            for i, block_item in enumerate(blockList):
                if "pList" in block_item:
                    pList = block_item["pList"]
                    block_params = {}

                    for param_item in pList:
                        param_key = param_item["param"]
                        param_value = param_item["value"]
                        if param_value is not None:
                            if param_key in ["IN_DIMENSION", "OUT_DIMENSION", "IN_CHANNEL", "OUT_CHANNEL", "KERNEL_SIZE", "STRIDE", "PADDING"]:
                                param_value = int(param_value)
                            block_params[param_key] = param_value

                    if "blockId" in block_item:
                        block_id = block_item["blockId"]
                        if block_id == 1009:
                            params.append({"param": "LAYER_TYPE[{}]".format(layer_count), "value": "Fc"})
                        elif block_id == 1010:
                            params.append({"param": "LAYER_TYPE[{}]".format(layer_count), "value": "Conv"})

                        if "ACTIVATION_FUN" in block_params:
                            params.append({"param": "LAYER_ARGS[{}]['ACTIVATION_FUN']".format(layer_count), "value": block_params["ACTIVATION_FUN"]})
                        if "IN_DIMENSION" in block_params:
                            params.append({"param": "LAYER_ARGS[{}]['IN_DIMENSION']".format(layer_count), "value": block_params["IN_DIMENSION"]})
                        if "OUT_DIMENSION" in block_params:
                            params.append({"param": "LAYER_ARGS[{}]['OUT_DIMENSION']".format(layer_count), "value": block_params["OUT_DIMENSION"]})
                        if "IN_CHANNEL" in block_params:
                            params.append({"param": "LAYER_ARGS[{}]['IN_CHANNEL']".format(layer_count), "value": block_params["IN_CHANNEL"]})
                        if "OUT_CHANNEL" in block_params:
                            params.append({"param": "LAYER_ARGS[{}]['OUT_CHANNEL']".format(layer_count), "value": block_params["OUT_CHANNEL"]})
                        if "KERNEL_SIZE" in block_params:
                            params.append({"param": "LAYER_ARGS[{}]['KERNEL_SIZE']".format(layer_count), "value": block_params["KERNEL_SIZE"]})
                        if "STRIDE" in block_params:
                            params.append({"param": "LAYER_ARGS[{}]['STRIDE']".format(layer_count), "value": block_params["STRIDE"]})
                        if "PADDING" in block_params:
                            params.append({"param": "LAYER_ARGS[{}]['PADDING']".format(layer_count), "value": block_params["PADDING"]})
                        if "POOL" in block_params:
                            params.append({"param": "LAYER_ARGS[{}]['POOL']".format(layer_count), "value": block_params["POOL"]})

                        layer_count += 1
    return params


def generate_json(target_path, args):
    data = args.get('data')['algorithmList']
    params = extract_params(data)
    json_data = json.dumps(params, indent=4)
    target_path = target_path[0:-12] #修正路径名称

    if os.path.exists(target_path+"generated.json"):
        os.remove(target_path+"generated.json")

    with open(target_path+"generated.json", "w") as file:
        file.write(json_data)


if __name__ == '__main__':
    pass