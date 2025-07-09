#!/usr/bin/python3

network_to_n_trials = {
    "resnet_152": 27000,
    "resnet_50": 27000,
    "wide_resnet_50": 27000,
    "googlenet": 27000,
    "densenet_121": 72000,
    "alexnet": 72000,
    "vgg_16": 72000,
    "resnet_18": 18000,
    "resnext_50": 27000,
    "inception_v3": 55000,
    "mobilenet_v2": 32000,
    "mobilenet_v3": 52000,
}

networks_dict = {
    "resnet_18",
    "resnet_50",
    "resnext_50",
    "wide_resnet_50",
    "mobilenet_v2",
    "mobilenet_v3",
    "resnet_152",
    "inception_v3",
    "alexnet",
    "densenet_121",
    "vgg_16",
    "googlenet",
}

def get_networks_arg():
    result = []
    for network in networks_dict:
        result.append(network)
    result.append("all")
    return result
