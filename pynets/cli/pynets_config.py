#!/usr/bin/env python3
"""
Created on Tue Nov  7 10:40:07 2017
Copyright (C) 2017
"""
import sys
import pkg_resources
import yaml

yaml.preserve_quotes = True


def load_runconfig():
    """
    Load the run configuration from a file.
    """
    with open(
        pkg_resources.resource_filename("pynets", "advanced.yaml"), mode="r+"
    ) as stream:
        advanced_params = yaml.load(stream, Loader=yaml.FullLoader)
    stream.close()
    return advanced_params


def save_runconfig(advanced_params: str):
    """
    Save the run configuration to a file.
    """
    with open(
        pkg_resources.resource_filename("pynets", "advanced.yaml"), mode="w+"
    ) as stream:
        yaml.dump(advanced_params, stream)
    stream.close()
    return


def nested_replace(d, key, new_value):
    """
    Replace a nested key in a dictionary.
    """
    for k, v in d.items():
        if isinstance(v, dict):
            nested_replace(v, key, new_value)
        if isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    nested_replace(item, key, new_value)
        if k == key:
            d[k] = new_value
    return


if __name__ == "__main__":
    advanced_params = load_runconfig()
    # argument 1: Configuration parameter
    # argument 2: Updated value
    nested_replace(advanced_params, sys.argv[0], sys.argv[1])
    save_runconfig(advanced_params)
