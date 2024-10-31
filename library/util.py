#!/usr/bin/python3

network_to_n_trials = {
    "resnet_50": 27000,
    "resnet_18": 18000,
    "mobilenet_v2": 32000,
    "mobilenet_v3": 52000,
    "wide_resnet_50": 27000,
    "resnext_50": 27000,
    "resnet3d_18": 17000,
    "inception_v3": 55000,
    "densenet_121": 72000,
    "bert": 9000,
    "dcgan": 5000,
}

networks_dict = {
    "resnet_18",
    "resnet_50",
    "resnet_152",
    "mobilenet_v2",
    "mobilenet_v3",
    "wide_resnet_50",
    "resnext_50",
    "resnet3d_18",
    "inception_v3",
    "densenet_121",
    "bert",
    "dcgan",
    "googlenet",
    "alexnet",
    "vgg_16"
}

def get_networks_arg():
    result = []
    for network in networks_dict:
        result.append(network)
    result.append("all")
    return result
