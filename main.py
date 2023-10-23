import sys
from argparse import ArgumentParser

from pytorch_lightning import Trainer

from lightning.entity_data_module import EntityDataModule
from lightning.entity_determined_logger import EntityDeterminedLogger
from lightning.entity_loop import EntityLoop
from models.lstm import LSTM
from models.transformer import Transformer
from models.vae import VAE
from models.gan import GAN
from models.tranAD import TranAD

from utils.namespace import augment_namespace_with_yaml, namespace_to_list


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
        args = augment_namespace_with_yaml(args, args.config)

    # Will this code be executed in a Determined environment?
    parser.add_argument("--determined", action="store_true")

    # Add trainer arguments
    parser.add_argument("--max_epochs", type=int, required=True)

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
    parser = EntityDataModule.add_argparse_args(parser)

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
    data_module = EntityDataModule.from_argparse_args(args)

    # ------------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------------
    model = model_cls(in_channels=data_module.dataset.dimension, **vars(args))

    # ------------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------------
    # Create the trainer
    trainer = Trainer(max_epochs=args.max_epochs, log_every_n_steps=20, accelerator="auto")

    # If the runtime is Determined AI, use `DeterminedLogger`
    if args.determined:
        trainer.logger = EntityDeterminedLogger()

    # Add the entity fit loop
    original_fit_loop = trainer.fit_loop
    custom_fit_loop = EntityLoop(len(data_module.dataset.entities))
    custom_fit_loop.connect(original_fit_loop)
    trainer.fit_loop = custom_fit_loop

    # Fit the model
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
