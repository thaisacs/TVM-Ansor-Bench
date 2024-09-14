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

        input_shape = args[0]

        input_data = torch.randn(input_shape).type(dtype2torch(dtype))
        scripted_model = torch.jit.trace(model, input_data).eval()

        input_name = "input0"
        shape_list = [(input_name, input_shape)]
        mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
        mod = convert_to_nhwc(mod)
        inputs = [(input_name, input_shape, dtype)]
    elif name == "bert":
        import gluonnlp

        input_info = args[0]
        model_name = input_info[0]
        batch_size = input_info[1]
        seq_length = input_info[2]

        # Instantiate a BERT classifier using GluonNLP
        dataset = "book_corpus_wiki_en_uncased"
        model, _ = gluonnlp.model.get_model(
            name=model_name,
            dataset_name=dataset,
            pretrained=True,
            use_pooler=True,
            use_decoder=False,
            use_classifier=False,
        )

        # Convert the MXNet model into TVM Relay format
        shape_dict = {
            "data0": (batch_size, seq_length),
            "data1": (batch_size, seq_length),
            "data2": (batch_size,),
        }
        mod, params = relay.frontend.from_mxnet(model, shape_dict)
        input_shape = (shape_dict["data0"], shape_dict["data1"], shape_dict["data2"])
        input_name = "input0"
        #dtype = 'int64'

        inputs = [(input_name, input_shape, dtype)]

        mod = tvm.relay.transform.FastMath()(mod)
        mod = tvm.relay.transform.EliminateCommonSubexpr()(mod)
        BindPass = tvm.relay.transform.function_pass(
            lambda fn, new_mod, ctx: tvm.relay.build_module.bind_params_by_name(
                fn, params
            ),
            opt_level=1,
        )
        mod = BindPass(mod)
        mod = tvm.relay.transform.FoldConstant()(mod)
        mod = tvm.relay.transform.CombineParallelBatchMatmul()(mod)
        mod = tvm.relay.transform.FoldConstant()(mod)
    elif name == "dcgan":
        output_shape = args[0]
        batch_size = output_shape[0]
        oshape = output_shape[1:]
        mod, params = relay.testing.dcgan.get_workload(
            batch_size=batch_size, oshape=oshape, layout="NHWC"
        )
        inputs = [("data", (100,), dtype)]
    else:
        raise ValueError("Invalid name: " + name)

    return mod, params, inputs

def build_network_keys():
    network_keys = []

    # resnet_18
    for batch_size in [1, 16, 32, 64, 128]:
        for image_size in [224, 240, 256]:
            for layer in [18]:
                network_keys.append((f'resnet_{layer}',
                                    [(batch_size, 3, image_size, image_size)]))

    # resnet_50
    for batch_size in [1, 16, 32, 64, 128]:
        for image_size in [224, 240, 256]:
            for layer in [50]:
                network_keys.append((f'resnet_{layer}',
                                    [(batch_size, 3, image_size, image_size)]))

    # mobilenet_v2
    for batch_size in [1, 16, 32, 64, 128]:
        for image_size in [224, 240, 256]:
            for name in ['mobilenet_v2']:
                network_keys.append((f'{name}',
                                    [(batch_size, 3, image_size, image_size)]))

    # mobilenet_v3
    for batch_size in [1, 16, 32, 64, 128]:
        for image_size in [224, 240, 256]:
            for name in ['mobilenet_v3']:
                network_keys.append((f'{name}',
                                    [(batch_size, 3, image_size, image_size)]))

    # wide-resnet
    for batch_size in [1, 16, 32, 64, 128]:
        for image_size in [224, 240, 256]:
            for layer in [50]:
                network_keys.append((f'wide_resnet_{layer}',
                                    [(batch_size, 3, image_size, image_size)]))

    # resnext
    for batch_size in [1, 16, 32, 64, 128]:
        for image_size in [224, 240, 256]:
            for layer in [50]:
                network_keys.append((f'resnext_{layer}',
                                    [(batch_size, 3, image_size, image_size)]))

    # inception-v3
    for batch_size in [1, 16, 32, 64, 128]:
        for image_size in [299]:
            network_keys.append((f'inception_v3',
                                [(batch_size, 3, image_size, image_size)]))

    # densenet
    for batch_size in [1, 16, 32, 64, 128]:
        for image_size in [224, 240, 256]:
            network_keys.append((f'densenet_121',
                                [(batch_size, 3, image_size, image_size)]))

    # resnet3d
    for batch_size in [1, 16, 32, 64, 128]:
        for image_size in [112, 128, 144]:
            for layer in [18]:
                network_keys.append((f'resnet3d_{layer}',
                                    [(batch_size, 3, image_size, image_size, 16)]))

    # bert 12
    for batch_size in [1, 16, 32, 64, 128]:
        for seq_length in [64, 128, 256]:
            for name in ["bert_12_768_12"]:
                network_keys.append((f'bert',
                                    [(name, batch_size, seq_length)]))

    # bert 24
    for batch_size in [1, 16, 32, 64, 128]:
        for seq_length in [64, 128, 256]:
            for name in ["bert_24_1024_16"]:
                network_keys.append((f'bert',
                                    [(name, batch_size, seq_length)]))

    # dcgan
    for batch_size in [1, 16, 32, 64, 128]:
        for image_size in [64]:
            network_keys.append((f'dcgan',
                                [(batch_size, 3, image_size, image_size)]))

    return network_keys

