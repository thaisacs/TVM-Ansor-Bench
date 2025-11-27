from os import walk
import os
import json
import numpy as np 
import statistics 
from scipy.stats import sem

# -----------------------------------------------------------------------------------

def gen_cached(filec):
    cache_miss = 0
    cache_hit = 0
    h_file  = open(filec, "r")
    for line in h_file:
        if('cache info: miss' in line):
            cache_miss += line.count('cache info: miss')
        if('cache info: hit' in line):
            cache_hit += line.count('cache info: hit')
    return cache_miss, cache_hit

if __name__ == "__main__":
    dir_base = "/home/thais.camacho/benchs/TVM-Ansor-Bench/"
    filesc_ = [
        ["AlexNet",       "tmp_logs/autoscheduler/llvm/search_space_model_1000/end-to-end-cache-results/cache-results-01/alexnet/slurm-alexnet.out"],
        ["DenseNet121",   "tmp_logs/autoscheduler/llvm/search_space_model_1000/end-to-end-cache-results/cache-results-01/densenet_121/slurm-densenet_121.out"],
        ["GoogleNet",     "tmp_logs/autoscheduler/llvm/search_space_model_1000/end-to-end-cache-results/cache-results-01/googlenet/slurm-googlenet.out"],
        ["InceptionV3",   "tmp_logs/autoscheduler/llvm/search_space_model_1000/end-to-end-cache-results/cache-results-01/inception_v3/slurm-inception_v3.out"],
        ["MobileNetV2",   "tmp_logs/autoscheduler/llvm/search_space_model_1000/end-to-end-cache-results/cache-results-01/mobilenet_v2/slurm-mobilenet_v2.out"],
        ["MobileNetV3",   "tmp_logs/autoscheduler/llvm/search_space_model_1000/end-to-end-cache-results/cache-results-01/mobilenet_v3/slurm-mobilenet_v3.out"],
        ["ResNet18",      "tmp_logs/autoscheduler/llvm/search_space_model_1000/end-to-end-cache-results/cache-results-01/resnet_18/slurm-resnet_18.out"],
        ["ResNet50",      "tmp_logs/autoscheduler/llvm/search_space_model_1000/end-to-end-cache-results/cache-results-01/resnet_50/slurm-resnet_50.out"],
        ["ResNet152",     "tmp_logs/autoscheduler/llvm/search_space_model_1000/end-to-end-cache-results/cache-results-01/resnet_152/slurm-resnet_152.out"],
        ["ResNeXt50",     "tmp_logs/autoscheduler/llvm/search_space_model_1000/end-to-end-cache-results/cache-results-01/resnext_50/slurm-resnext_50.out"],
        ["VGG16",         "tmp_logs/autoscheduler/llvm/search_space_model_1000/end-to-end-cache-results/cache-results-01/vgg_16/slurm-vgg_16.out"],
        ["WideResNet50",  "tmp_logs/autoscheduler/llvm/search_space_model_1000/end-to-end-cache-results/cache-results-01/wide_resnet_50/slurm-wide_resnet_50.out"]
    ]

    print("name,value,type")

    for x in filesc_:
        path = dir_base + x[1]
        cache_miss, cache_hit = gen_cached(path)
        total = cache_miss + cache_hit
        #print(cache_miss, cache_hit, total)
        hit_ratio = 100 * (cache_hit / total)
        print(x[0] + "," + str(hit_ratio) + ",TGC-Ansor")
