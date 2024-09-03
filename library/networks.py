#!/usr/bin/python3

import torch
import torchvision.models as models  # torchvision>=0.9.0
import transformers  # pip3 install transformers==3.5 torch==1.7
import tvm.relay.testing
from common import (
    convert_to_nhwc,
    dtype2torch,
    NETWORK_INFO_FOLDER,
    get_relay_ir_filename,
    get_task_info_filename,
)
from tvm import relay
import os

def get_network_with_key(network_key):
    name, args = network_key
    args = args[0]

    if name in [
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
    ]:
        if name in ["resnet_18", "resnet_50"]:
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

        # input_shape = args[0]
        input_shape = args[0][0]
        dtype = "float32"

        input_data = torch.randn(input_shape).type(dtype2torch(dtype))
        scripted_model = torch.jit.trace(model, input_data).eval()

        input_name = "input0"
        shape_list = [(input_name, input_shape)]
        mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
        mod = convert_to_nhwc(mod)
        inputs = [(input_name, input_shape, dtype)]
    elif name in ["bert_squeeze"]:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        model = transformers.SqueezeBertForSequenceClassification.from_pretrained(
            "squeezebert/squeezebert-uncased", return_dict=False
        )

        # input_shape = args[0]
        input_shape = args[0][0]

        input_shape = input_shape
        input_name = "input_ids"
        input_dtype = "int64"
        A = torch.randint(10000, input_shape)

        model.eval()
        scripted_model = torch.jit.trace(model, [A], strict=False)

        input_name = "input_ids"
        shape_list = [(input_name, input_shape)]
        mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

        mod = relay.transform.FastMath()(mod)
        mod = relay.transform.CombineParallelBatchMatmul()(mod)

        inputs = [(input_name, input_shape, input_dtype)]
    elif name == "dcgan":
        # output_shape = args[0]
        output_shape = args[0][0]
        batch_size = output_shape[0]
        oshape = output_shape[1:]
        mod, params = relay.testing.dcgan.get_workload(
            batch_size=batch_size, oshape=oshape, layout="NHWC"
        )
        inputs = [("data", (100,), "float32")]
    else:
        raise ValueError("Invalid name: " + name)

    return mod, params, inputs

def get_network(network_args):
    name, batch_size = network_args["network"], network_args["batch_size"]
    if name in [
        "resnet_18",
        "resnet_50",
        "mobilenet_v2",
        "mobilenet_v3",
        "wide_resnet_50",
        "resnext_50",
        "densenet_121",
        "vgg_16",
        "resnet3d_18",
    ]:
        network_key = (name, [(batch_size, 3, 224, 224)])
    elif name in ["inception_v3"]:
        network_key = (name, [(batch_size, 3, 299, 299)])
    elif name in [
        "bert_tiny",
        "bert_base",
        "bert_medium",
        "bert_large",
        "bert_squeeze",
    ]:
        network_key = (name, [(batch_size, 128)])
    elif name == "dcgan":
        network_key = (name, [(batch_size, 3, 64, 64)])
    else:
        raise ValueError("Invalid network: " + name)

    return get_network_with_key(network_key)

def build_network_pairs_keys():
    network_keys = []

    # resnext
    batch_size = 1
    image_size = 224
    layer = 50

    network_keys.append((f"resnext_{layer}", [(batch_size, 3, image_size, image_size)]))

    # inception-v3
    batch_size=1
    image_size=299
    network_keys.append(
       (f"inception_v3", [(batch_size, 3, image_size, image_size)])
    )

    return network_keys

def build_network_keys():
    network_keys = []

    # resnext
    for batch_size in [1]:
        for image_size in [224, 240, 256]:
            for layer in [50]:
                network_keys.append(
                    (f"resnext_{layer}", [(batch_size, 3, image_size, image_size)])
                )

    # inception-v3
    for batch_size in [1, 2, 4]:
        for image_size in [299]:
            network_keys.append(
                (f"inception_v3", [(batch_size, 3, image_size, image_size)])
            )

    # densenet
    for batch_size in [1, 2, 4]:
        for image_size in [224, 240, 256]:
            network_keys.append(
                (f"densenet_121", [(batch_size, 3, image_size, image_size)])
            )

    # resnet_18 and resnet_50
    for batch_size in [1, 4, 8]:
        for image_size in [224, 240, 256]:
            for layer in [18, 50]:
                network_keys.append(
                    (f"resnet_{layer}", [(batch_size, 3, image_size, image_size)])
                )

    # mobilenet_v2
    for batch_size in [1, 4, 8]:
        for image_size in [224, 240, 256]:
            for name in ["mobilenet_v2", "mobilenet_v3"]:
                network_keys.append(
                    (f"{name}", [(batch_size, 3, image_size, image_size)])
                )

    # resnet3d
    for batch_size in [1, 4, 8]:
        for image_size in [112]:
            for layer in [18]:
                network_keys.append(
                    (f"resnet3d_{layer}", [(batch_size, 3, image_size, image_size, 16)])
                )

    # bert
    for batch_size in [1, 2, 4]:
        for seq_length in [64, 128, 256]:
            for scale in ["squeeze"]:
                # for scale in ['tiny', 'base', 'medium', 'large']:
                network_keys.append((f"bert_{scale}", [(batch_size, seq_length)]))

    # dcgan
    for batch_size in [1, 4, 8]:
        for image_size in [64]:
            network_keys.append((f"dcgan", [(batch_size, 3, image_size, image_size)]))

    # wide-resnet
    for batch_size in [1, 4]:
        for image_size in [224, 240, 256]:
            for layer in [50]:
                network_keys.append(
                    (f"wide_resnet_{layer}", [(batch_size, 3, image_size, image_size)])
                )

    return network_keys

def main():
    for name in build_network_keys():
        print(name)

    networks = build_network_keys()

    for arg in networks:
        network_arg = {
            "network": arg[0],
            "batch_size": arg[1],
        }
        mod, params, inputs = get_network(network_arg)

main()