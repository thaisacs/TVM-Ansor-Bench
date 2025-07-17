#!/usr/bin/python3

import torch
import torchvision.models as models

import onnx
import torch.onnx

import tvm
from tvm.relay.frontend.onnx import from_onnx

# -------------------------

def load_model(model_name: str, use_weights=True):
    try:
        # Get constructor from torchvision.models
        constructor = getattr(models, model_name)

        if use_weights:
            # Create weights enum name, e.g. "ResNet18_Weights"
            weights_enum_name = f"{model_name.capitalize()}_Weights"

            # Fix for camelCase models (e.g., efficientnet_b0)
            weights_enum = getattr(models, weights_enum_name, None)
            if weights_enum is not None:
                weights = weights_enum.DEFAULT
                return constructor(weights=weights).eval()

        # If no weights requested or not found
        return constructor().eval()

    except AttributeError:
        raise ValueError(f"Model '{model_name}' is not available in torchvision.models")

def get_network_with_key(network_key, dtype):
    name = network_key["network"]
    name_replaced = network_key["network"].replace("_", "")

    if name == "mobilenet_v2":
        name_replaced = name
    elif name == "mobilenet_v3":
        name_replaced = name + "_large"
    elif name == "wide_resnet_50":
        name_replaced = "wide_resnet50_2"
    elif name == "resnext_50":
        name_replaced = "resnext50_32x4d"
    elif name == "inception_v3":
        name_replaced = name

    torch_model = load_model(name_replaced)
    torch.onnx.export(
        torch_model,
        network_key["args"],
        name_replaced + ".onnx",
        opset_version=14,
        input_names=["input"],
        output_names=["output"],
    )
    onnx_model = onnx.load(name_replaced + ".onnx")
    mod, params = from_onnx(
        onnx_model,
        freeze_params=False,
    )
    #mod, params = relay.frontend.detach_params(mod)
    return mod, params

def build_network_keys():
    network_keys = []

    # googlenet
    for batch_size in [1]:
        for image_size in [224]:
            args = torch.randn(
                batch_size, 3, image_size, image_size, dtype=torch.float32
            )
            network_keys.append((f"googlenet", args))

    # alexnet
    for batch_size in [1]:
        for image_size in [224]:
            args = torch.randn(
                batch_size, 3, image_size, image_size, dtype=torch.float32
            )
            network_keys.append((f"alexnet", args))

    # vgg
    for batch_size in [1]:
        for image_size in [224]:
            args = torch.randn(
                batch_size, 3, image_size, image_size, dtype=torch.float32
            )
            network_keys.append((f"vgg_16", args))

    # resnet_18
    for batch_size in [1]:
        for image_size in [224]:
            args = torch.randn(
                batch_size, 3, image_size, image_size, dtype=torch.float32
            )
            network_keys.append((f"resnet_18", args))

    # resnet_50
    for batch_size in [1]:
        for image_size in [224]:
            args = torch.randn(
                batch_size, 3, image_size, image_size, dtype=torch.float32
            )
            network_keys.append((f"resnet_50", args))

    # resnet_152
    for batch_size in [1]:
        for image_size in [224]:
            args = torch.randn(
                batch_size, 3, image_size, image_size, dtype=torch.float32
            )
            network_keys.append((f"resnet_152", args))

    # mobilenet_v2
    for batch_size in [1]:
        for image_size in [224]:
            args = torch.randn(
                batch_size, 3, image_size, image_size, dtype=torch.float32
            )
            network_keys.append((f"mobilenet_v2", args))

    # mobilenet_v3
    for batch_size in [1]:
        for image_size in [224]:
            args = torch.randn(
                batch_size, 3, image_size, image_size, dtype=torch.float32
            )
            network_keys.append((f"mobilenet_v3", args))

    # wide-resnet
    for batch_size in [1]:
        for image_size in [224]:
            args = torch.randn(
                batch_size, 3, image_size, image_size, dtype=torch.float32
            )
            network_keys.append((f"wide_resnet_50", args))

    # resnext
    for batch_size in [1]:
        for image_size in [224]:
            args = torch.randn(
                batch_size, 3, image_size, image_size, dtype=torch.float32
            )
            network_keys.append((f"resnext_50", args))

    # inception-v3
    for batch_size in [1]:
        for image_size in [299]:
            args = torch.randn(
                batch_size, 3, image_size, image_size, dtype=torch.float32
            )
            network_keys.append((f"inception_v3", args))

    # densenet
    for batch_size in [1]:
        for image_size in [224]:
            args = torch.randn(
                batch_size, 3, image_size, image_size, dtype=torch.float32
            )
            network_keys.append((f"densenet_121", args))

    return network_keys
