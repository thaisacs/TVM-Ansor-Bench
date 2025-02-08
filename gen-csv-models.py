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
                approache = dir_path.split('/')[10].split('-')[3]
                if(approache == 'cache'):
                    approache = 'TGC'
                else:
                    approache = 'TVM'
                idx = dir_path.split('/')[10].split('-')
                if(len(idx) == 5):
                    idx = idx[4]
                else:
                    approache = approache + '-ES'
                    idx = idx[5]
                net = filename.split('.')[0].split('-')[1]
                time = get_tuning_time(dir_path+'/'+filename)
                if(idx == '01'):
                    if(not net in arr):
                        arr[net] = {}
                    arr[net][approache] = time
                    #print(approache + ',' + idx + ',' + net + ',' + str(time))

def get_tuning_time(filename):
    with open(filename) as f:
        for line in f:
            pass
    last_line = line
    value = float(last_line.split(' ')[2].replace('\n', ''))
    return value

if __name__ == "__main__":
    filesc_ = [
        "/home/thais/Dev/TVMBench/tmp_logs/autoscheduler/llvm/model_tuning_space/end-to-end-cache/",
        "/home/thais/Dev/TVMBench/tmp_logs/autoscheduler/llvm/model_tuning_space/end-to-end-original/",
        "/home/thais/Dev/TVMBench/tmp_logs/autoscheduler/llvm/model_tuning_space/end-to-end-cache-threshold/",
        "/home/thais/Dev/TVMBench/tmp_logs/autoscheduler/llvm/model_tuning_space/end-to-end-original-threshold/"
    ]

    print('approach, id, model_name, tuning_time')
    arr = {}
    for x in filesc_:
        gen_file_out(x, arr)

    for net in arr:
        #print(net + ',' + str(arr[net]['TVM']/arr[net]['TVM']) + ',' + str(arr[net]['TGC']/arr[net]['TVM']) + ',' + str(arr[net]['TVM-ES']/arr[net]['TVM']) + ',' + str(arr[net]['TGC-ES']/arr[net]['TVM']))
        print('TVM,0' + ',' + net + ',' + str(arr[net]['TVM']/arr[net]['TVM']))
        print('TGC,0' + ',' + net + ',' + str(arr[net]['TGC']/arr[net]['TVM']))
        print('TVM-ES,0' + ',' + net + ',' + str(arr[net]['TVM-ES']/arr[net]['TVM']))
        print('TGC-ES,0' + ',' + net + ',' + str(arr[net]['TGC-ES']/arr[net]['TVM']))


    #count = 0
    #for x in filesc_:
    #    values = []
    #    for y in x[1]:
    #        v = get_tuning_time(y)
    #        values.append(v)

    #    s1 = values[0]/values[0]
    #    s2 = values[1]/values[0]
    #    s3 = values[2]/values[0]
    #    print(x[0]+','+str(values[0])+','+str(values[1])+','+str(values[2])+','+str(s1)+','+str(s2)+','+str(s3))
