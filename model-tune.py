#!/usr/bin/python3

import argparse
import os
import time

from library.networks import get_network_with_key, build_network_keys
from tvm.driver import tvmc
from library.util import get_networks_arg, networks_dict, network_to_n_trials

import tvm
from tvm import relay, auto_scheduler

# --------------------------------------------------------------------------------------------

def auto_scheduler_tune(network_arg, dtype, target, log_file, tune):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    if os.path.exists(log_file):
        os.remove(log_file)

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

    tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)
    for idx, task in enumerate(tasks):
        print(
            "========== Task %d  (workload key: %s) =========="
            % (idx, task.workload_key)
        )
        print(task.compute_dag)

    if(tune):
        tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
        start = time.time()
        tuner.tune(
                tuning_opt,
                #per_task_early_stopping=64*5,
                subgraph_cache="/home/thais.camacho/tvm/src/auto_cache/params.yaml"
        )
        end = time.time()
        print(end - start)

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
                print("Tune %s ..." % network_arg)

                log_file = os.path.join(
                    args.logdir, "autoscheduler", str(target.kind), str(network_arg) + ".json"
                )
                
                auto_scheduler_tune(network_arg, args.dtype, target, log_file, args.tune)

