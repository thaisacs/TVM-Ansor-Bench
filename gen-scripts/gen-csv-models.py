import matplotlib.pyplot as plt
from os import walk
import os
import json
import numpy as np 
import yaml
import statistics 
from scipy.stats import sem

# -----------------------------------------------------------------------------------

def gen_cached_hashs(filec):
    h_file  = open(filec, "r")
    tasks = []
    for line in h_file:
        if('#hash' in line):
            tasks.append(line.replace('\n', '').split(' '))
    return tasks

def gen_min(paths):
    min_dic = {}
    for dic in paths:
        for path in dic['path']:
            for (dir_path, dir_names, file_names) in walk(path):
                count = 0
                for filename in file_names:
                    print(filename)
                    if("output" not in filename):
                        count += 1
                        with open(os.path.join(dir_path, filename), 'r') as f:
                            for l in f:
                                result = json.loads(l)
                                task = filename + '+' + result['i'][0][0]
                                r = sum(result['r'][0])/len(result['r'][0])
                                if(task not in min_dic):
                                    min_dic[task] = r
                                elif(min_dic[task] > r):
                                    min_dic[task] = r
    return min_dic

def gen_arr_best(name, path):
    arr = []
    for (dir_path, dir_names, file_names) in walk(path):
        count = 0
        for filename in file_names:
            if("output" not in filename):
                count += 1
                with open(os.path.join(dir_path, filename), 'r') as f:
                    best = 1000
                    values = []
                    for l in f:
                        result = json.loads(l)
                        task = filename + '+' + result['i'][0][0]
                        r = sum(result['r'][0])/len(result['r'][0])
                        if(r < best):
                            best = r
                        if(best < 1000):
                            values.append(best)
                            
                    while(len(values) < 1000 and len(values) >= 800):
                        v = values[len(values)-1]
                        values.append(v)
                    if(len(values) >= 1000):
                        arr.append(values[:1000])

    return arr


def gen_arr_acc(name, path):
    arr = []
    for (dir_path, dir_names, file_names) in walk(path):
        count = 0
        for filename in file_names:
            if("output" not in filename):
                count += 1
                with open(os.path.join(dir_path, filename), 'r') as f:
                    _sum = 0
                    values = []
                    for l in f:
                        result = json.loads(l)
                        task = filename + '+' + result['i'][0][0]
                        r = sum(result['r'][0])
                        if(r < 1000):
                            _sum += r
                        values.append(_sum)

                    if(len(values) >= 1000 and sum(values)):
                        arr.append(values[:1000])
    return arr

def gen_best_it(path):
    arr = {}
    for (dir_path, dir_names, file_names) in walk(path):
        for filename in file_names:
            if("output" not in filename):
                with open(os.path.join(dir_path, filename), 'r') as f:
                    values = []
                    for l in f:
                        result = json.loads(l)
                        task = filename + '+' + result['i'][0][0]
                        r = sum(result['r'][0])/len(result['r'][0])
                        values.append(r)
                    if(min(values) != 10000000000):
                        arr[task] = [values.index(min(values)), min(values)]
    return arr

def gen_file_out(path, arr):
    for (dir_path, dir_names, file_names) in walk(path):
        for filename in file_names:
            if(".out" in filename):
                approache = dir_path.split('/')[10].split('-')[0]
                if(approache == 'cache'):
                    approache = 'TGC-Ansor'
                else:
                    approache = 'TVM-Ansor'
                net = filename.split('.')[0].split('-')[1]
                time = get_tuning_time(dir_path+'/'+filename)
                if(not net in arr):
                    arr[net] = {}
                    arr[net][approache] = [time]
                else:
                    if(not approache in arr[net]):
                        arr[net][approache] = [time]
                    else:
                        arr[net][approache].append(time)

def get_tuning_time(filename):
    with open(filename) as f:
        for line in f:
            if("tunning time:" in line):
                line = line.replace("\n", "")
                line = line.split(" ")[2]
                break
    last_line = line
    value = float(last_line)/60
    return value

if __name__ == "__main__":
    filesc_ = [
        "/home/thais.camacho/benchs/TVM-Ansor-Bench/tmp_logs/autoscheduler/llvm/search_space_model_1000/end-to-end-cache-results/",
        "/home/thais.camacho/benchs/TVM-Ansor-Bench/tmp_logs/autoscheduler/llvm/search_space_model_1000/end-to-end-original-results/"
    ]

    print('approach, model_name, tuning_mean, tuning_std')
    arr = {}
    for x in filesc_:
        gen_file_out(x, arr)

    for net in arr:
        speedups_1 = []
        speedups_2 = []
        for i in range(0, len(arr[net]['TVM-Ansor'])):
            speedups_1.append(arr[net]['TGC-Ansor'][i])
            speedups_2.append(arr[net]['TVM-Ansor'][i])

        if(net == "alexnet"):
            netf = "AlexNet"
        if(net == "resnet_18"):
            netf = "ResNet18"
        if(net == "resnet_50"):
            netf = "ResNet50"
        if(net == "resnext_50"):
            netf = "ResNeXt50"
        if(net == "wide_resnet_50"):
            netf = "WideResNet50"
        if(net == "mobilenet_v2"):
            netf = "MobileNetV2"
        if(net == "mobilenet_v3"):
            netf = "MobileNetV3"
        if(net == "resnet_152"):
            netf = "ResNet152"
        if(net == "inception_v3"):
            netf = "InceptionV3"
        if(net == "densenet_121"):
            netf = "DenseNet121"
        if(net == "vgg_16"):
            netf = "VGG16"
        if(net == "googlenet"):
            netf = "GoogleNet"

        print('TGC-Ansor,' + netf + ',' + str(np.mean(speedups_1)) + ',' + str(np.std(speedups_1)))
        print('TVM-Ansor,' + netf + ',' + str(np.mean(speedups_2)) + ',' + str(np.std(speedups_2)))
