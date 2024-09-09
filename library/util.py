#!/usr/bin/python3

network_to_n_trials = {
    #"resnet_50": 22000,
    #"mobilenet_v2": 16000,
    #"inception_v3": 26000,
    #"bert": 12000,
    "resnet_50": 10000,
    "mobilenet_v2": 10000,
    "inception_v3": 10000,
    "bert": 10000,
    "resnet_18": 10000,
    "resnet_50": 10000,
    "mobilenet_v2": 10000,
    "mobilenet_v3": 10000,
    "wide_resnet_50": 10000,
    "resnext_50": 10000,
    "resnet3d_18": 10000,
    "inception_v3": 10000,
    "densenet_121": 10000,
    "vgg_16": 10000,
    "bert": 10000,
    "dcgan": 10000,
}

networks_dict = {
    "resnet_18",
    "resnet_50",
    "mobilenet_v2",
    "mobilenet_v3",
    "wide_resnet_50",
    "resnext_50",
    "resnet3d_18",
    "inception_v3",
    "densenet_121",
    "vgg_16",
    "bert",
    "dcgan"
}

def get_networks_arg():
    result = []
    for network in networks_dict:
        result.append(network)
    result.append("all")
    return result

