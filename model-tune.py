#!/usr/bin/python3

from library.networks import get_network_with_key, build_network_keys
from tvm.driver import tvmc

if __name__ == "__main__":
    networks = build_network_keys()

    for arg in networks:
        print(arg)
        network_arg = {
            "network": arg[0],
            "batch_size": arg[1],
        }
        mod, params, inputs = get_network_with_key(network_arg)
        model = tvmc.TVMCModel(mod, params)

