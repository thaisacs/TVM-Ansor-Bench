import matplotlib.pyplot as plt
from os import walk
import os
import json
import numpy as np 
import yaml
import statistics 

# -----------------------------------------------------------------------------------

if __name__ == "__main__":
    mypath = '/home/thais/Dev/TVMBench/tmp_logs/autoscheduler/llvm/origin-results/output-densenet_121-O01'
    filenames = next(walk(mypath), (None, None, []))[2]  # [] if no file

    print('iteration,' + 'value,' + 'desvio_padrao')
    arr = []
    for (dir_path, dir_names, file_names) in walk(mypath):
        count = 0
        for filename in file_names:
            if("output" not in filename):
                count += 1
                with open(os.path.join(dir_path, filename), 'r') as f:
                    iteration = 0
                    best = 100000
                    values = []
                    for l in f:
                        if(iteration == 1000):
                            break
                        result = json.loads(l)
                        task = result['i'][0][0]
                        r = sum(result['r'][0])/len(result['r'][0])
                        if(r < best):
                            best = r
                        if(best < 1000):
                            values.append(best)
                        iteration += 1

                    if(len(values) == 1000):
                        fmin = min(values)
                        for idx, v in enumerate(values):
                            values[idx] = values[idx] / fmin 

                        arr.append(values)
    #x = []
    #y = []

    #fmin = 100000
    #for idx, v in enumerate(arr):
    #    for n in v:
    #        if(n < fmin):
    #            fmin = n

    #for idx, v in enumerate(arr):
    #    for idy, n in enumerate(v):
    #        arr[idx][idy] = n/fmin
    
    #for idx, v in enumerate(arr):
    #    x.append(statistics.mean(arr[idx]))
    #    y.append(statistics.stdev(arr[idx]))
    #for idx, v in enumerate(x):
    #    print(str(idx+1)+', '+str(v)+', '+str(y[idx]))

    for idx in range(0, 1000):
        l = []
        for v in arr:
            l.append(v[idx])
        print(str(idx+1)+', '+str(statistics.geometric_mean(l)))
