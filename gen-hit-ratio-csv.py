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

def gen_arr_best(paths, name, min_dict):
    arr = []
    for path in paths:
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
                            fmin = min_dict[task]
                            for idx, v in enumerate(values):
                                values[idx] = values[idx] / fmin 
                            arr.append(values[:1000])

    for idx in range(0, 1000):
        l = []
        for v in arr:
            l.append(v[idx])
        print(str(idx)+','+str(statistics.mean(l))+','+str(sem(l))+','+name)


def gen_arr_acc(paths, name, min_dict):
    arr = []
    for path in paths:
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
                                fmin = min_dict[task]
                                _sum += r / fmin
                            values.append(_sum)

                        if(len(values) >= 1000):
                            arr.append(values[:1000])

    for idx in range(0, 1000):
        l = []
        for v in arr:
            l.append(v[idx])
        print(str(idx)+','+str(statistics.mean(l))+','+str(sem(l))+','+name)

def gen_paths(path):
    arr = []
    for (dir_path, dir_names, file_names) in walk(path):
        if('output-' in dir_path):
            arr.append(dir_path)
    return arr        
    #paths.append({'path': original_arr, 'name': 'original'})
    #return paths

if __name__ == "__main__":
    filesc_ = [
    ["alexnet",        "/home/thais/Dev/TVMBench/tmp_logs/autoscheduler/llvm/search_space_1000/cache-results-01/output-alexnet-M01-sort/output-alexnet.out"],
    ["densenet_121",   "/home/thais/Dev/TVMBench/tmp_logs/autoscheduler/llvm/search_space_1000/cache-results-01/output-densenet_121-M01-sort/output-densenet121.out"],
    ["googlenet",      "/home/thais/Dev/TVMBench/tmp_logs/autoscheduler/llvm/search_space_1000/cache-results-01/output-googlenet-M01-sort/output-googlenet.out"],
    ["inception_v3",   "/home/thais/Dev/TVMBench/tmp_logs/autoscheduler/llvm/search_space_1000/cache-results-01/output-inception_v3-M01-sort/output-inception_v3.out"],
    ["mobilenet_v2",   "/home/thais/Dev/TVMBench/tmp_logs/autoscheduler/llvm/search_space_1000/cache-results-01/output-mobilenet_v2-M01-sort/output-mobilenet_v2.out"],
    ["mobilenet_v3",   "/home/thais/Dev/TVMBench/tmp_logs/autoscheduler/llvm/search_space_1000/cache-results-01/output-mobilenet_v3-M01-sort/output-mobilenet_v3.out"],
    ["resnet_18",      "/home/thais/Dev/TVMBench/tmp_logs/autoscheduler/llvm/search_space_1000/cache-results-01/output-resnet_18-M01-sort/output-resnet_18.out"],
    ["resnet_50",      "/home/thais/Dev/TVMBench/tmp_logs/autoscheduler/llvm/search_space_1000/cache-results-01/output-resnet_50-M01-sort/output-resnet_50.out"],
    ["resnet_152",     "/home/thais/Dev/TVMBench/tmp_logs/autoscheduler/llvm/search_space_1000/cache-results-01/output-resnet_152-M01-sort/output-resnet_152.out"],
    ["resnext_50",     "/home/thais/Dev/TVMBench/tmp_logs/autoscheduler/llvm/search_space_1000/cache-results-01/output-resnext_50-M01-sort/output-resnext_50.out"],
    ["vgg_16",         "/home/thais/Dev/TVMBench/tmp_logs/autoscheduler/llvm/search_space_1000/cache-results-01/output-vgg_16-M01-sort/output-vgg_16.out"],
    ["wide_resnet_50", "/home/thais/Dev/TVMBench/tmp_logs/autoscheduler/llvm/search_space_1000/cache-results-01/output-wide_resnet_50-M01-sort/output-wide_resnet_50.out"]
    ]

    print("name,value")

    for x in filesc_:
        tasks = gen_cached_hashs(x[1])

        cache_miss_hit = len(tasks)
        cache_miss = 0
        for task in tasks:
            if(int(task[2]) == 0):
                cache_miss += 1
    
        print(x[0] + "," + str(100*((cache_miss_hit-cache_miss)/cache_miss_hit)))
