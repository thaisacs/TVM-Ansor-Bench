from os import walk
import os
import json
import statistics 
from scipy.stats import sem

# -----------------------------------------------------------------------------------

def gen_min(info):
    min_dic = {}
    for dic in info:
        count = 0
        for filename in dic['filenames']:
            count += 1
            with open(filename, 'r') as f:
                for l in f:
                    result = json.loads(l)
                    task = result['i'][0][0]
                    r = sum(result['r'][0])/len(result['r'][0])
                    if(task not in min_dic):
                        min_dic[task] = r
                    elif(min_dic[task] > r):
                        min_dic[task] = r
    return min_dic

def gen_arr_best(filenames, name, min_dict):
    arr = []
    count = 0
    for filename in filenames:
        count += 1
        with open(filename, 'r') as f:
            best = 1000
            values = []
            for l in f:
                result = json.loads(l)
                task = result['i'][0][0]
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

def gen_arr_acc(filenames, name, min_dict):
    count = 0
    arr = []
    for filename in filenames:
        count += 1
        with open(filename, 'r') as f:
            _sum = 0
            values = []
            for l in f:
                result = json.loads(l)
                task = result['i'][0][0]
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
        for filename in file_names:
            if(".json" in filename):
                arr.append(os.path.join(dir_path, filename))
    return arr        

if __name__ == "__main__":
    paths = []
    cache_path = '/home/thais.camacho/benchs/TVM-Ansor-Bench/tmp_logs/autoscheduler/llvm/search_space_task_1000/cache-results/cache-results-01/'
    cache_arr = gen_paths(cache_path)
    paths.append({'filenames': cache_arr, 'name': 'TGC-Ansor'})
    original_path = '/home/thais.camacho/benchs/TVM-Ansor-Bench/tmp_logs/autoscheduler/llvm/search_space_task_1000/original-results/original-results-01/'
    original_arr = gen_paths(original_path)
    paths.append({'filenames': original_arr, 'name': 'TVM-Ansor'})

    min_dict = gen_min(paths)

    print('iteration,' + 'value,' + 'desvio,' + 'tipo')
    for x in paths:
        #gen_arr_best(x['filenames'], x['name'], min_dict)
        gen_arr_acc(x['filenames'], x['name'], min_dict)
