import inspect
from argparse import Namespace
from typing import List

import yaml


def augment_arguments_with_yaml(namespace: Namespace, path: str) -> Namespace:
    """
    Augment arguments with the arguments stored in a YAML file.

    :param namespace: The argument namespace to augment.
    :param path: The path the YAML file.
    :return: The augmented argument namespace.
    """
    yaml_dict = yaml.load(open(path), Loader=yaml.FullLoader)
    yaml_dict.update(vars(namespace))
    return Namespace(**yaml_dict)


def namespace_to_list(namespace: Namespace) -> List[str]:
    """
    Convert an argument namespace into a list of command line arguments.

    :param namespace: The argument namespace to convert.
    :return: The list of command line arguments.
    """
    arg_list = []
    for k, v in vars(namespace).items():
        arg_list.append("--" + k)
        arg_list.append(str(v))
    return arg_list


def create_from_arguments(cls: any, namespace: Namespace, **kwargs) -> any:
    # Get the constructor argument names (as list)
    arg_names = inspect.getfullargspec(cls.__init__).args

    # Remove "self" from the list
    arg_names.remove("self")

    # Get the arguments according to the list
    filtered_args = dict(filter(lambda i: i[0] in arg_names, vars(namespace).items()))

    # Augment the arguments with kwargs
    filtered_args.update(kwargs)

    return cls(**filtered_args)
