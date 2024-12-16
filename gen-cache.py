import matplotlib.pyplot as plt
from os import walk
import os
import json
import numpy as np 
import yaml

# -----------------------------------------------------------------------------------

def sort_array(arr):
    #return reversed(arr)
    for idx, result in enumerate(arr):
        key = sum(result['r'][0])/len(result['r'][0])
        j = idx - 1

        keyb = sum(arr[j]['r'][0])/len(arr[j]['r'][0])
        while j >= 0 and key < keyb:
            arr[j + 1] = arr[j]
            j -= 1
            keyb = sum(arr[j]['r'][0])/len(arr[j]['r'][0])
        arr[j + 1] = result
    return arr

def sort_search_space(search_space):
    for hashx in search_space:
        search_space[hashx] = sort_array(search_space[hashx])

def dump_to_file(search_space):
    task_id = 0
    files_in_cache = {}
    for idx, taskx in enumerate(search_space):
        shape = taskx[38:]
        h = taskx[2:34]

        shape = taskx[37:len(taskx)-1]
        h = taskx[2:34]

        configs = []
        for x in search_space[taskx]:
            configs.append(str(x))

        data = {'id': task_id, 'hash': h, 'shape': shape, 'space': configs}

        if(not h in files_in_cache):
            files_in_cache[h] = []

        data = {'hash': h, 'shape': shape, 'space': configs}

        #with open('cachex'+str(task_id)+'.yml', 'w') as outfile:
        #    yaml.dump(data, outfile)
        files_in_cache[h].append('cachex'+str(task_id)+'.yml')
        
        task_id += 1
        #with open('cachex'+str(task_id)+'.yml', 'w') as outfile:
        #    yaml.dump(data, outfile)
    
    for h in files_in_cache:
        data = {'size': len(files_in_cache[h]), 'files': files_in_cache[h]}
        with open(h+'.yml', 'a') as outfile:
            yaml.dump(data, outfile)

if __name__ == "__main__":
    mypath_origin = '/home/thais/Dev/TVMBench/tmp_logs/autoscheduler/llvm/search_space_1000/original-results-01/'
    filenames_origin = next(walk(mypath_origin), (None, None, []))[2]  # [] if no file
    res = []
    lines = 0
    search_space = {}

    for (dir_path, dir_names, file_names) in walk(mypath_origin):
            res.extend(file_names)
            for filename in file_names:
                if("output" not in filename):
                    #if('mobilenet_v2' == net.replace('\'', '') or 'mobilenet_v3' == net.replace('\'', '')):
                    #    print(net)
                    #if('densenet_121' == net.replace('\'', '') or 'alexnet' == net.replace('\'', '')):
                    #    print(net)
                    #if('vgg_16' == net.replace('\'', '') or 'googlenet' == net.replace('\'', '')):
                    #    print(net)
                    #if('resnet_50' == net.replace('\'', '') or 'resnet_18' == net.replace('\'', '')):
                    net = filename.split(" ")[1].replace(',', '')
                    if('resnext_50' == net.replace('\'', '') or 'wide_resnet_50' == net.replace('\'', '')):
                        ...
                    else:
                        with open(os.path.join(dir_path, filename), 'r') as f:
                            for l in f:
                                result = json.loads(l)
                                task = result['i'][0][0]
                                hashx = task
                                if(hashx not in search_space):
                                    search_space[hashx] = []
                                search_space[hashx].append(result)
                        
    #sort_search_space(search_space)
    dump_to_file(search_space)

    #for hashx in search_space:
    #    print(hashx)
    #    for result in search_space[hashx]:
    #        r = sum(result['r'][0])/len(result['r'][0])
    #        print(r)
