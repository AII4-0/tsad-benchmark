from argparse import Namespace
from typing import List

import yaml


def augment_namespace_with_yaml(namespace: Namespace, path: str) -> Namespace:
    """
    Augment the namespace with the arguments stored in a YAML file.

    :param namespace: The namespace to augment.
    :param path: The path the YAML file.
    :return: The augmented namespace.
    """
    yaml_dict = yaml.load(open(path), Loader=yaml.FullLoader)
    yaml_dict.update(vars(namespace))
    return Namespace(**yaml_dict)


def namespace_to_list(namespace: Namespace) -> List[str]:
    """
    Convert a namespace into a list of command line arguments.

    :param namespace: The namespace to convert.
    :return: The list of command line arguments.
    """
    arg_list = []
    for k, v in vars(namespace).items():
        arg_list.append("--" + k)
        arg_list.append(str(v))
    return arg_list
