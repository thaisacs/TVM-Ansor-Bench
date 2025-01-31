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

def model_run(network_arg, dtype, target, log_file):
    mod, params, inputs = get_network_with_key(network_arg, dtype)

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
        for x in range(0, 1):
            print(module.benchmark(dev, repeat=10, number=10, min_repeat_ms=500, end_to_end=True))

    np.random.seed(0)
    data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
    run_bench(inputs[0][0], data_tvm, lib)

    np.random.seed(0)
    data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
    actual_output1 = get_output(inputs[0][0], data_tvm, lib)
    expected_output = get_output(inputs[0][0], data_tvm, ref_lib)

    tvm.testing.assert_allclose(actual_output1, expected_output, rtol=1e-4, atol=1e-4)

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
        "--logfile", type=str, default="tmp_logs/log.json", help="Log filename."
    )
    args = parser.parse_args()

    if args.network == "all":
        networks = networks_dict
    else:
        networks = [args.network]
    dtypes = [args.dtype]

    target = tvm.target.Target(args.target)

    networks_keys = build_network_keys()

    for arg in networks_keys:
        network = arg[0]
        if(network in networks):
            network_arg = {
                "network": arg[0],
                "args": arg[1],
            }
            model_run(network_arg, args.dtype, target, args.logfile)
