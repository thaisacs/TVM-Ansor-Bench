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

def auto_scheduler_tune(network_arg, dtype, target, tune):
    mod, params, inputs = get_network_with_key(network_arg, dtype)
    n_trials = 1000

    tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)
    for idx, task in enumerate(tasks):
        print(
            "========== Task %d  (workload key: %s) =========="
            % (idx, task.workload_key)
        )
        print(task.compute_dag)

    if(tune):
        for task in tasks:
            log_file = os.path.join(
                args.logdir, "autoscheduler", str(target.kind), str(network_arg) + str(task.workload_key) + ".json"
            )
            
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            if os.path.exists(log_file):
                os.remove(log_file)

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

            # Run auto-tuning (search)
            start = time.time()
            task.tune(tuning_opt)
            end = time.time()
            print("task tune: ", str(task.workload_key), end - start)


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
            print("Tune %s ..." % network_arg)

            auto_scheduler_tune(network_arg, args.dtype, target, args.tune)

