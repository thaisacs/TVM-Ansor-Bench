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

def get_it(path, best_it):
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
                        for i, v in enumerate(values):
                            if(v <= best_it[task][1]):
                                arr[task] = [i, v]
                                break
    return arr

if __name__ == "__main__":
    filesc_ = [
    ["alexnet",        ["/home/thais/Dev/TVMBench/tmp_logs/autoscheduler/llvm/search_space_1000/cache-results-01/output-alexnet-M01-sort/",
                        "/home/thais/Dev/TVMBench/tmp_logs/autoscheduler/llvm/search_space_1000/original-results-01/output-alexnet-O01/"]],
    ["densenet_121",   ["/home/thais/Dev/TVMBench/tmp_logs/autoscheduler/llvm/search_space_1000/cache-results-01/output-densenet_121-M01-sort/",
                        "/home/thais/Dev/TVMBench/tmp_logs/autoscheduler/llvm/search_space_1000/original-results-01/output-densenet_121-O01/"]],
    ["googlenet",      ["/home/thais/Dev/TVMBench/tmp_logs/autoscheduler/llvm/search_space_1000/cache-results-01/output-googlenet-M01-sort/",
                        "/home/thais/Dev/TVMBench/tmp_logs/autoscheduler/llvm/search_space_1000/original-results-01/output-googlenet-O01/"]],
    ["inception_v3",   ["/home/thais/Dev/TVMBench/tmp_logs/autoscheduler/llvm/search_space_1000/cache-results-01/output-inception_v3-M01-sort/",
                        "/home/thais/Dev/TVMBench/tmp_logs/autoscheduler/llvm/search_space_1000/original-results-01/output-inception_v3-O01/"]],
    ["mobilenet_v2",   ["/home/thais/Dev/TVMBench/tmp_logs/autoscheduler/llvm/search_space_1000/cache-results-01/output-mobilenet_v2-M01-sort/",
                        "/home/thais/Dev/TVMBench/tmp_logs/autoscheduler/llvm/search_space_1000/original-results-01/output-mobilenet_v2-O01/"]],
    ["mobilenet_v3",   ["/home/thais/Dev/TVMBench/tmp_logs/autoscheduler/llvm/search_space_1000/cache-results-01/output-mobilenet_v3-M01-sort/",
                        "/home/thais/Dev/TVMBench/tmp_logs/autoscheduler/llvm/search_space_1000/original-results-01/output-mobilenet_v3-O01/"]],
    ["resnet_18",      ["/home/thais/Dev/TVMBench/tmp_logs/autoscheduler/llvm/search_space_1000/cache-results-01/output-resnet_18-M01-sort/",
                        "/home/thais/Dev/TVMBench/tmp_logs/autoscheduler/llvm/search_space_1000/original-results-01/output-resnet_18-O01"]],
    ["resnet_50",      ["/home/thais/Dev/TVMBench/tmp_logs/autoscheduler/llvm/search_space_1000/cache-results-01/output-resnet_50-M01-sort/",
                        "/home/thais/Dev/TVMBench/tmp_logs/autoscheduler/llvm/search_space_1000/original-results-01/output-resnet_50-O01/"]],
    ["resnet_152",     ["/home/thais/Dev/TVMBench/tmp_logs/autoscheduler/llvm/search_space_1000/cache-results-01/output-resnet_152-M01-sort/",
                        "/home/thais/Dev/TVMBench/tmp_logs/autoscheduler/llvm/search_space_1000/original-results-01/output-resnet_152-O01/"]],
    ["resnext_50",     ["/home/thais/Dev/TVMBench/tmp_logs/autoscheduler/llvm/search_space_1000/cache-results-01/output-resnext_50-M01-sort/",
                        "/home/thais/Dev/TVMBench/tmp_logs/autoscheduler/llvm/search_space_1000/original-results-01/output-resnext_50-O01/"]],
    ["vgg_16",         ["/home/thais/Dev/TVMBench/tmp_logs/autoscheduler/llvm/search_space_1000/cache-results-01/output-vgg_16-M01-sort/",
                        "/home/thais/Dev/TVMBench/tmp_logs/autoscheduler/llvm/search_space_1000/original-results-01/output-vgg_16-O01/"]],
    ["wide_resnet_50", ["/home/thais/Dev/TVMBench/tmp_logs/autoscheduler/llvm/search_space_1000/cache-results-01/output-wide_resnet_50-M01-sort/",
                        "/home/thais/Dev/TVMBench/tmp_logs/autoscheduler/llvm/search_space_1000/original-results-01/output-wide_resnet_50-O01/"]]
    ]

    #for x in filesc_:
    #    best_it = gen_best_it(x[1][1])
    #    best = get_it(x[1][0], best_it)
    #    print(len(best_it), len(best))
    #    for task in best_it:
    #        print(best_it[task])
    #    print()
    #    for task in best:
    #        print(best[task])

    print('name,value,sd,sem')

    count = 0
    for x in filesc_:
        #arr_original = gen_arr_best(x[0], x[1][1])
        #arr_cache = gen_arr_best(x[0], x[1][0])
        #arr = []
        #for it, value in enumerate(arr_original):
        #    arr.append(arr_original[it][len(arr_original[it]) - 1]/arr_cache[it][350])
        #    #arr.append(arr_original[it][350]/arr_cache[it][350])
        
        #for v in arr:
        #    print(x[0]+','+str(v))
        ##print(x[0]+','+str(statistics.mean(arr))+','+str(statistics.stdev(arr))+','+str(sem(arr)))

        arr_original = gen_arr_acc(x[0], x[1][1])
        arr_cache = gen_arr_acc(x[0], x[1][0])
        arr = []
        for it, value in enumerate(arr_original):
            arr.append(arr_original[it][350]/arr_cache[it][350])
            #arr.append(arr_original[it][len(arr_original[it]) - 1]/arr_cache[it][350])
            #arr.append(arr_original[it][len(arr_original[it]) - 1]/arr_cache[it][len(arr_cache[it] )- 1])
        ##for v in arr:
        ##    print(str(count)+','+str(v))
        ##    count = count + 1
            
        print(x[0]+','+str(statistics.mean(arr))+','+str(statistics.stdev(arr))+','+str(sem(arr)))