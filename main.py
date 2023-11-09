import sys
from argparse import ArgumentParser

from benchmark.benchmark import Benchmark
from benchmark.data_module import DataModule
from models.gan import GAN
from models.lstm import LSTM
from models.tran_ad import TranAD
from models.transformer import Transformer
from models.vae import VAE
from utils.arguments import augment_arguments_with_yaml, namespace_to_list, create_from_arguments


def main() -> None:
    """Execute the whole program."""
    # ------------------------------------------------------------------------
    # Manage args
    # ------------------------------------------------------------------------
    # Create the argument parser
    parser = ArgumentParser()

    # Arguments from YAML configuration file ?
    parser.add_argument("--config", type=str)

    # Parse the known arguments
    args, _ = parser.parse_known_args()

    # If the arguments are stored in YAML configuration file, load them
    if args.config:
        args = augment_arguments_with_yaml(args, args.config)

    # Add benchmark arguments
    parser = Benchmark.add_argparse_args(parser)

    # Which model?
    parser.add_argument(
        "--model",
        choices=[
            "LSTM",
            "Transformer",
            "VAE",
            "GAN",
            "TranAD"
        ],
        required=True
    )

    # Add data module arguments
    parser = DataModule.add_argparse_args(parser)

    # Create a list of arguments
    list_of_args = namespace_to_list(args) + sys.argv[1:]

    # Parse the known arguments
    args, _ = parser.parse_known_args(args=list_of_args)

    # Get the model class
    if args.model == "LSTM":
        model_cls = LSTM
    elif args.model == "Transformer":
        model_cls = Transformer
    elif args.model == "VAE":
        model_cls = VAE
    elif args.model == "GAN":
        model_cls = GAN
    elif args.model == "TranAD":
        model_cls = TranAD
    else:
        raise RuntimeError("Unrecognized model")

    # Add model specific arguments
    parser = model_cls.add_argparse_args(parser)

    # Parse all arguments
    args = parser.parse_args(args=list_of_args)

    # ------------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------------
    # Create the data module
    data_module = create_from_arguments(DataModule, args)

    # ------------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------------
    model = create_from_arguments(model_cls, args, in_channels=data_module.dataset.dimension)

    # ------------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------------
    # Create the benchmark
    benchmark = create_from_arguments(Benchmark, args)

    # Benchmark the model
    benchmark.run(model, data_module)


if __name__ == "__main__":
    main()
