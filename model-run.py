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

# --------------------------------------------------------------------------------------------

def auto_scheduler_run(network_arg, dtype, target):
    mod, params, inputs = get_network_with_key(network_arg, dtype)
    n_trials = network_to_n_trials[network_arg['network']]

    if "cpu" in target.keys:
        tuning_opt = auto_scheduler.TuningOptions(
            num_measure_trials=n_trials,
            runner=auto_scheduler.LocalRunner(repeat=10, enable_cpu_cache_flush=True),
            measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        )
    else:
        min_repeat_ms = 450 if network in ["bert"] else 300
        measure_ctx = auto_scheduler.LocalRPCMeasureContext(
            repeat=1, min_repeat_ms=min_repeat_ms, timeout=10
        )
        tuning_opt = auto_scheduler.TuningOptions(
            num_measure_trials=n_trials,
            runner=measure_ctx.runner,
            measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        )

    print("Compile...")
    input_shape = (224, 224, 3)
    output_shape = (1, 1000)

    log_file = "/home/thais.camacho/TVMBench/cache.json"
    with auto_scheduler.ApplyHistoryBest(log_file):
        with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
            lib = relay.build(mod, target=target, params=params)
    
    # Create graph executor
    dev = tvm.device(str(target), 0)
    module = graph_executor.GraphModule(lib["default"](dev))
    data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
    module.set_input(inputs[0][0], data_tvm)

    # Evaluate
    print("Evaluate inference time cost...")
    for x in range(0, 5):
        print(module.benchmark(dev, repeat=10, min_repeat_ms=500, end_to_end=True))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TVM Model Tune.\n')
    parser.add_argument(
        "--network",
        type=str,
        choices=get_networks_arg(),
        default="all",
        help="The name of the neural network.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="llvm",
        help="The compilation target.",
    )
    parser.add_argument("--dtype", type=str, default="float32", help="The data type.")
    parser.add_argument("--tune", help="The tune activate flag.", action='store_true')
    parser.add_argument(
        "--logdir", type=str, default="tmp_logs/", help="Log file directory."
    )
    parser.add_argument("--shape", type=int, default=1, help="The input shape.")
    args = parser.parse_args()

    if args.network == "all":
        networks = networks_dict
        shape_idx = -1
    else:
        networks = [args.network]
        shape_idx = 1

    dtypes = [args.dtype]

    target = tvm.target.Target(args.target)

    networks_keys = build_network_keys()

    for arg_idx, arg in enumerate(networks_keys):
        if(shape_idx != -1):
            if(arg_idx > 0 and networks_keys[arg_idx-1][0] == arg[0]):
                shape_idx += 1
            else:
                shape_idx = 1

        if(shape_idx == -1 or shape_idx == args.shape):
            network = arg[0]
            if(network in networks):
                network_arg = {
                    "network": arg[0],
                    "args": arg[1],
                }
                auto_scheduler_run(network_arg, args.dtype, target)

