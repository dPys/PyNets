#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""A convenience tool for querying PyNets's version."""
import sys
import os.path as op


def main():
    """Print current PyNets version."""
    sys.path.insert(0, op.abspath('.'))
    from pynets.__about__ import __version__
    print(__version__)


if __name__ == '__main__':
    main()
