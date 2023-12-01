from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import tensorflow as tf
import torch
from sklearn.metrics import mean_squared_error

from benchmark.data_module import DataModule
from utils import constants


def main() -> None:
    """The main function of the script."""

    # Create the argument parser
    # noinspection DuplicatedCode
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--dataset", choices=constants.DATASET_NAMES, required=True)
    parser.add_argument("--entity", type=int, required=True)
    parser.add_argument("--window_size", type=int, required=True)
    parser.add_argument("--pytorch_model", type=Path, required=True)
    parser.add_argument("--tflite_model", type=Path, required=True)
    parser.add_argument("--start_index_inputs_exported_in_c", type=int, default=0)
    parser.add_argument("--n_inputs_exported_in_c", type=int, default=0)

    # Parse all arguments
    args = parser.parse_args()

    # Create the data module
    data_module = DataModule(args.data_dir, args.dataset, args.window_size, 1, args.start_index_inputs_exported_in_c, args.n_inputs_exported_in_c)

    # Prepare data
    data_module.prepare_data()

    # Get the test dataloader for the entity
    y, test_dataloader = data_module[args.entity]

    # Disable gradient calculation
    torch.autograd.set_grad_enabled(False)

    # Load the PyTorch model
    pytorch_model = torch.load(args.pytorch_model)
    pytorch_model.eval()

    # Load the TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=str(args.tflite_model.absolute()))
    interpreter.allocate_tensors()

    # Get the TFLite input/output tensor index
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    # Create lists to store the predictions
    pytorch_predictions = []
    tflite_predictions = []

    # Iterate over test batches
    i_input = 0
    for x, y in test_dataloader:
        # Perform the prediction with the PyTorch model
        x = x[:, : -1]
        # y = torch.tensor([[[0.0]]])
        # pytorch_pred = pytorch_model(x, y)
        pytorch_pred = pytorch_model(x)

        # Perform the prediction with the TFLite model
        interpreter.set_tensor(input_index, x.numpy())
        interpreter.invoke()
        tflite_pred = interpreter.get_tensor(output_index)

        # Add predictions to the lists
        pytorch_predictions.append(pytorch_pred.numpy().squeeze())
        tflite_predictions.append(tflite_pred.squeeze())

        # Print input data
        if i_input >=400 and i_input < 500:
            print(f'Input i: {i_input} data : {x}')

        i_input = i_input + 1

    # Convert lists to tensors
    pytorch_predictions = np.array(pytorch_predictions)
    tflite_predictions = np.array(tflite_predictions)

    # Compute the mean squared error
    mae = mean_squared_error(pytorch_predictions.flatten(), tflite_predictions.flatten())

    # Logging
    print(f"Mean squared error: {mae:.20f}")
    print(f'Pytorch prediction {pytorch_predictions}')
    print(f'Tflite prediction {tflite_predictions}')
    print(f'Tflite prediction[0:100] {tflite_predictions[0:100]}')
    print(f'Tflite prediction[400:500] {tflite_predictions[400:500]}')
    pass


if __name__ == "__main__":
    main()
