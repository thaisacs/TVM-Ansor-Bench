#!/usr/bin/python3

import argparse
import os

from library.networks import get_network_with_key, build_network_keys
from tvm.driver import tvmc
from library.util import get_networks_arg, networks_dict, network_to_n_trials

import tvm
from tvm import relay, auto_scheduler

import sys

import numpy as np

from tvm.relay import data_dep_optimization as ddo
import tvm.relay.testing
from tvm.contrib import graph_executor

import matplotlib.pyplot as plt
from os import walk
import os
import json
import numpy as np 
import yaml
import statistics 
from scipy.stats import sem


# --------------------------------------------------------------------------------------------

def model_run(network_arg, dtype, target, log_file):
    mod, params, inputs = get_network_with_key(network_arg, dtype)

    #print("Compile...")
    #input_shape = (3, 224, 224)

    #with auto_scheduler.ApplyHistoryBest(log_file):
    #    with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
    #        lib = relay.build(mod, target=target, params=params)
    
    ## Create graph executor
    #dev = tvm.device(str(target), 0)
    #module = graph_executor.GraphModule(lib["default"](dev))
    #data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
    #module.set_input(inputs[0][0], data_tvm)

    ## Evaluate
    #print("Evaluate inference time cost...")
    #for x in range(0, 1):
    #    print(module.benchmark(dev, repeat=10, number=10, min_repeat_ms=500, end_to_end=True))
    print("Compile...")
    input_shape = inputs[0][1]

    with auto_scheduler.ApplyHistoryBest(log_file):
        with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
            lib = relay.build(mod, target=target, params=params)
            
    with auto_scheduler.ApplyHistoryBest(log_file):
        with tvm.transform.PassContext(opt_level=0, config={"relay.backend.use_auto_scheduler": True}):
            ref_lib = relay.build(mod, target=target, params=params)

    # Check the correctness
    def get_output(input_data, data, lib):
        dev = tvm.device(str(target), 0)
        module = graph_executor.GraphModule(lib["default"](dev))
        module.set_input(input_data, data)
        module.run()
        return module.get_output(0).numpy()

    def run_bench(input_data, data, lib):
        dev = tvm.device(str(target), 0)
        # Create graph executor
        module = graph_executor.GraphModule(lib["default"](dev))
        module.set_input(input_data, data)
        # Evaluate
        print("Evaluate inference time cost...")
        for x in range(0, 10):
            print(module.benchmark(dev, repeat=10, number=10, min_repeat_ms=500, end_to_end=True))

    np.random.seed(0)
    data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
    run_bench(inputs[0][0], data_tvm, lib)

    np.random.seed(0)
    data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
    actual_output1 = get_output(inputs[0][0], data_tvm, lib)
    expected_output = get_output(inputs[0][0], data_tvm, ref_lib)

    tvm.testing.assert_allclose(actual_output1, expected_output, rtol=1e-4, atol=1e-4)

#if __name__ == "__main__":
#    parser = argparse.ArgumentParser(description='TVM Model Tune.\n')
#    parser.add_argument(
#        "--network",
#        type=str,
#        choices=get_networks_arg(),
#        default="all",
#        help="The name of the neural network.",
#    )
#    parser.add_argument(
#        "--target",
#        type=str,
#        default="llvm",
#        help="The compilation target.",
#    )
#    parser.add_argument("--dtype", type=str, default="float32", help="The data type.")
#    parser.add_argument("--tune", help="The tune activate flag.", action='store_true')
#    parser.add_argument(
#        "--logfile", type=str, default="tmp_logs/log.json", help="Log filename."
#    )
#    args = parser.parse_args()
#
#    if args.network == "all":
#        networks = networks_dict
#    else:
#        networks = [args.network]
#
#    target = tvm.target.Target(args.target)
#
#    networks_keys = build_network_keys()
#
#    for arg in networks_keys:
#        network = arg[0]
#        if(network in networks):
#            network_arg = {
#                "network": arg[0],
#                "args": arg[1],
#            }
#            model_run(network_arg, args.dtype, target, args.logfile)

def gen_file_out(path, arr):
    for (dir_path, dir_names, file_names) in walk(path):
        for filename in file_names:
            if(".json" in filename):
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
                net = filename.split(' ')[1]
                net = net.replace(',', '').replace('\'', '')
                if(idx == '01'):
                    if(not net in arr):
                        arr[net] = {}
                    arr[net][approache] = dir_path + '/' + filename

if __name__ == "__main__":
    networks_keys = build_network_keys()
    target = 'llvm'
    dtype = 'float32'

    filesc_ = [
        "/home/thais/Dev/TVMBench/tmp_logs/autoscheduler/llvm/model_tuning_space/end-to-end-cache/",
        "/home/thais/Dev/TVMBench/tmp_logs/autoscheduler/llvm/model_tuning_space/end-to-end-original/",
        "/home/thais/Dev/TVMBench/tmp_logs/autoscheduler/llvm/model_tuning_space/end-to-end-cache-threshold/",
        "/home/thais/Dev/TVMBench/tmp_logs/autoscheduler/llvm/model_tuning_space/end-to-end-original-threshold/"
    ]

    arr = {}
    for x in filesc_:
        gen_file_out(x, arr)

    for net in arr:
        networks = [net]

        if(net != "inception_v3"):

            for arg in networks_keys:
                network = arg[0]
                if(network in networks):
                    network_arg = {
                        "network": arg[0],
                        "args": arg[1],
                    }
                    for x in arr[net]:
                        logfile = arr[net][x]
                        print("#new_experiment")
                        print(net)
                        print(x)
                        model_run(network_arg, dtype, target, logfile)
