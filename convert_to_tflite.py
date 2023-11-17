from argparse import ArgumentParser
from pathlib import Path

import nobuco
import tensorflow as tf
import torch
from nobuco import ChannelOrder

from benchmark.data_module import DataModule
from utils import constants
import os

def convert_to_c(tflite_model, file_name, path0):
    from tensorflow.lite.python.util import convert_bytes_to_c_source
    source_text, header_text = convert_bytes_to_c_source(tflite_model, file_name)

    if not os.path.exists(path0):
        os.makedirs(path0) 

    with  open(os.path.join(path0, file_name + '.h'), 'w') as file:
        file.write(header_text)

    with  open(os.path.join(path0, file_name + '.cpp'), 'w') as file:
        file.write("\n#include \"" + file_name + ".h\"\n")
        file.write(source_text)

def main() -> None:
    """The main function of the script."""

    # Create the argument parser
    # noinspection DuplicatedCode
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--dataset", choices=constants.DATASET_NAMES, required=True)
    parser.add_argument("--entity", type=int, required=True)
    parser.add_argument("--window_size", type=int, required=True)
    parser.add_argument("--model", type=Path, required=True)

    # Parse all arguments
    args = parser.parse_args()

    # Create the data module
    data_module = DataModule(args.data_dir, args.dataset, args.window_size, 1)

    # Prepare data
    data_module.prepare_data()

    # Get the train dataloader for the entity
    train_dataloader, _ = data_module[args.entity]

    # Get a data sample
    data_sample = next(iter(train_dataloader))

    # Load the PyTorch model
    pytorch_model = torch.load(args.model)

    # Convert the PyTorch model to Keras model
    keras_model = nobuco.pytorch_to_keras(
        pytorch_model,
        args=[data_sample],
        inputs_channel_order=ChannelOrder.PYTORCH,
        outputs_channel_order=ChannelOrder.PYTORCH
    )

    # Convert the Keras model to TensorFlow lite
    tflite_model = tf.lite.TFLiteConverter.from_keras_model(keras_model).convert()

    # Save the model
    with open(args.model.parent.joinpath(args.model.stem + ".tflite"), "wb") as f:
        f.write(tflite_model)

    # convert to C source code and store it
    print("convert to C source code")
    source_file = args.model.stem
    convert_to_c(tflite_model, source_file, args.model.parent)



if __name__ == "__main__":
    main()
