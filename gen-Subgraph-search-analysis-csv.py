import matplotlib.pyplot as plt
from os import walk
import os
import json
import numpy as np 
import yaml
import statistics 
from scipy.stats import sem

# -----------------------------------------------------------------------------------

def gen_cached_hashs(logs):
    for log in logs:
        h_file  = open(log, "r")
        hashs_cached = []
        for line in h_file:
            if('myhash' in line):
                h = line.split(' ')[1].replace('\n', '')
            if('cache size' in line):
                if(len(line.split(' ')) < 4):
                    size = int(line.split(' ')[2].replace('\n', ''))
                    if(size > 0):
                        hashs_cached.append(h)
    return hashs_cached

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

if __name__ == "__main__":
    paths = []
    #cache_path = '/home/thais/Dev/TVMBench/tmp_logs/autoscheduler/llvm/search_space_1000/cache-results-01/'
    #cache_arr = gen_paths(cache_path)
    #paths.append({'path': cache_arr, 'name': 'TGC - Experiment 1'})
    #cache_path = '/home/thais/Dev/TVMBench/tmp_logs/autoscheduler/llvm/search_space_1000/cache-results-02/'
    #cache_arr = gen_paths(cache_path)
    #paths.append({'path': cache_arr, 'name': 'TGC - Experiment 2'})
    #cache_path = '/home/thais/Dev/TVMBench/tmp_logs/autoscheduler/llvm/search_space_1000/cache-results-03/'
    #cache_arr = gen_paths(cache_path)
    #paths.append({'path': cache_arr, 'name': 'TGC - Experiment 3'})
    #cache_path = '/home/thais/Dev/TVMBench/tmp_logs/autoscheduler/llvm/search_space_1000/cache-results-04/'
    #cache_arr = gen_paths(cache_path)
    #paths.append({'path': cache_arr, 'name': 'TGC - Experiment 4'})
    cache_path = '/home/thais/Dev/TVMBench/tmp_logs/autoscheduler/llvm/search_space_1000/cache-results-01/'
    cache_arr = gen_paths(cache_path)
    paths.append({'path': cache_arr, 'name': 'TGC'})

    #original_path = '/home/thais/Dev/TVMBench/tmp_logs/autoscheduler/llvm/search_space_1000/original-results-01/'
    #original_arr = gen_paths(original_path)
    #paths.append({'path': original_arr, 'name': 'Ansor - Experiment 1'})
    #original_path = '/home/thais/Dev/TVMBench/tmp_logs/autoscheduler/llvm/search_space_1000/original-results-02/'
    #original_arr = gen_paths(original_path)
    #paths.append({'path': original_arr, 'name': 'Ansor - Experiment 2'})
    #original_path = '/home/thais/Dev/TVMBench/tmp_logs/autoscheduler/llvm/search_space_1000/original-results-03/'
    #original_arr = gen_paths(original_path)
    #paths.append({'path': original_arr, 'name': 'Ansor - Experiment 3'})
    #original_path = '/home/thais/Dev/TVMBench/tmp_logs/autoscheduler/llvm/search_space_1000/original-results-04/'
    #original_arr = gen_paths(original_path)
    #paths.append({'path': original_arr, 'name': 'Ansor - Experiment 4'})
    original_path = '/home/thais/Dev/TVMBench/tmp_logs/autoscheduler/llvm/search_space_1000/original-results-01/'
    original_arr = gen_paths(original_path)
    paths.append({'path': original_arr, 'name': 'Ansor'})

    min_dict = gen_min(paths)
    print('iteration,' + 'value,' + 'desvio,' + 'tipo')
    for x in paths:
        #gen_arr_best(x['path'], x['name'], min_dict)
        gen_arr_acc(x['path'], x['name'], min_dict)
