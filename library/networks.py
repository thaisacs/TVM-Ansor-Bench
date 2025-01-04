#!/usr/bin/python3

import torch
import torchvision.models as models  # torchvision>=0.9.0
import transformers  # pip3 install transformers==3.5 torch==1.7
import tvm.relay.testing
from library.common import (
    convert_to_nhwc,
    dtype2torch,
    NETWORK_INFO_FOLDER,
    get_relay_ir_filename,
    get_task_info_filename,
)
from tvm import relay
import os

from tvm.driver import tvmc

def get_network_with_key(network_key, dtype):
    name = network_key['network']
    args = network_key['args']

    if name in [
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
        "vgg_16",
        "googlenet",
        "alexnet",
        "vgg"
    ]:
        if name in ["resnet_18", "resnet_50", "resnet_152"]:
            model = getattr(models, name.replace("_", ""))(pretrained=False)
        elif name == "wide_resnet_50":
            model = getattr(models, "wide_resnet50_2")(pretrained=False)
        elif name == "resnext_50":
            model = getattr(models, "resnext50_32x4d")(pretrained=False)
        elif name == "mobilenet_v2":
            model = getattr(models, name)(pretrained=False)
        elif name == "mobilenet_v3":
            model = getattr(models, name + "_large")(pretrained=False)
        elif name == "inception_v3":
            model = getattr(models, name)(
                pretrained=False, aux_logits=False, init_weights=True
            )
        elif name == "densenet_121":
            model = getattr(models, name.replace("_", ""))(pretrained=False)
        elif name == "resnet3d_18":
            model = models.video.r3d_18(pretrained=False)
        elif name == "vgg_16":
            model = getattr(models, name.replace("_", ""))(pretrained=False)
        elif name == "googlenet":
            model = getattr(models, name.replace("_", ""))(pretrained=False)
        elif name == "alexnet":
            model = getattr(models, name.replace("_", ""))(pretrained=False)

        input_shape = args[0]

        input_data = torch.randn(input_shape).type(dtype2torch(dtype))
        scripted_model = torch.jit.trace(model, input_data).eval()

        input_name = "input0"
        shape_list = [(input_name, input_shape)]
        mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
        mod = convert_to_nhwc(mod)
        inputs = [(input_name, input_shape, dtype)]
    else:
        raise ValueError("Invalid name: " + name)

    return mod, params, inputs

def build_network_keys():
    network_keys = []

    # googlenet
    for batch_size in [1]:
        for image_size in [224]:
            network_keys.append((f'googlenet',
                                [(batch_size, 3, image_size, image_size)]))

    # alexnet
    for batch_size in [1]:
        for image_size in [224]:
            network_keys.append((f'alexnet',
                                [(batch_size, 3, image_size, image_size)]))

    # vgg
    for batch_size in [1]:
        for image_size in [224]:
            network_keys.append((f'vgg_16',
                                [(batch_size, 3, image_size, image_size)]))
    # resnet_18
    for batch_size in [1]:
        for image_size in [224]:
            network_keys.append((f'resnet_18',
                                [(batch_size, 3, image_size, image_size)]))

    # resnet_50
    for batch_size in [1]:
        for image_size in [224]:
            network_keys.append((f'resnet_50',
                                [(batch_size, 3, image_size, image_size)]))

    # resnet_152
    for batch_size in [1]:
        for image_size in [224]:
            network_keys.append((f'resnet_152',
                                [(batch_size, 3, image_size, image_size)]))

    # mobilenet_v2
    for batch_size in [1]:
        for image_size in [224]:
            network_keys.append((f'mobilenet_v2',
                                [(batch_size, 3, image_size, image_size)]))

    # mobilenet_v3
    for batch_size in [1]:
        for image_size in [224]:
            network_keys.append((f'mobilenet_v3',
                                [(batch_size, 3, image_size, image_size)]))

    # wide-resnet
    for batch_size in [1]:
        for image_size in [224]:
            network_keys.append((f'wide_resnet_50',
                                [(batch_size, 3, image_size, image_size)]))

    # resnext
    for batch_size in [1]:
        for image_size in [224]:
            network_keys.append((f'resnext_50',
                                [(batch_size, 3, image_size, image_size)]))

    # inception-v3
    for batch_size in [1]:
        for image_size in [299]:
            network_keys.append((f'inception_v3',
                                [(batch_size, 3, image_size, image_size)]))

    # densenet
    for batch_size in [1]:
        for image_size in [224]:
            network_keys.append((f'densenet_121',
                                [(batch_size, 3, image_size, image_size)]))

    return network_keys
