import matplotlib.pyplot as plt
from os import walk
import os
import json
import numpy as np 
import yaml
import statistics 
from scipy.stats import sem

# -----------------------------------------------------------------------------------

def get_min(paths):
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
                                task = result['i'][0][0]
                                r = sum(result['r'][0])/len(result['r'][0])
                                if(task not in min_dic):
                                    min_dic[task] = r
                                elif(min_dic[task] > r):
                                    min_dic[task] = r
    return min_dic

def gen_arr(paths, name, min_dict):
    arr = []
    for path in paths:
        for (dir_path, dir_names, file_names) in walk(path):
            count = 0
            for filename in file_names:
                if("output" not in filename):
                    count += 1
                    with open(os.path.join(dir_path, filename), 'r') as f:
                        iteration = 0
                        best = 100000
                        _sum = 0
                        values = []
                        for l in f:
                            if(iteration == 1000):
                                break
                            result = json.loads(l)
                            task = result['i'][0][0]
                            r = sum(result['r'][0])/len(result['r'][0])
                            if(r < best):
                                best = r
                            #if(best < 10000):
                            #    values.append(best)
                            if(best < 1000):
                                #fmin = min_dict[task]
                                #_sum += r / fmin
                                values.append(best)
                            iteration += 1
                            
                        if(len(values) >= 1000):
                            #fmin = min(values)
                            fmin = min_dict[task]
                            for idx, v in enumerate(values):
                                values[idx] = values[idx] / fmin 
                            arr.append(values[:1000])

    for idx in range(0, 1000):
        l = []
        for v in arr:
            l.append(v[idx])
        print(str(idx)+','+str(statistics.mean(l))+','+str(sem(l))+','+name)

if __name__ == "__main__":
    paths = [
    {'path': ['/home/thais/Dev/TVMBench/tmp_logs/autoscheduler/llvm/origin-results/output-resnet18-O01',
              '/home/thais/Dev/TVMBench/tmp_logs/autoscheduler/llvm/origin-results/output-resnet50-O01'], 'name': 'origin'},
    {'path': ['/home/thais/Dev/TVMBench/tmp_logs/autoscheduler/llvm/myversion-results/output-resnet18-M01-1500',
              '/home/thais/Dev/TVMBench/tmp_logs/autoscheduler/llvm/myversion-results/output-resnet50-M01-1500'], 'name': 'cache'},
    {'path': ['/home/thais/Dev/TVMBench/tmp_logs/autoscheduler/llvm/myversion-results/output-resnet18-M01-1000-sort',
              '/home/thais/Dev/TVMBench/tmp_logs/autoscheduler/llvm/myversion-results/output-resnet50-M01-1000'], 'name': 'cache-sort'}
    ]

    min_dict = get_min(paths)

    print('iteration,' + 'value,' + 'desvio,' + 'tipo')
    for x in paths:
        gen_arr(x['path'], x['name'], min_dict)
