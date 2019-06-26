#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# @Author: dPys
# @Date:   2019-06-25 09:42:38
import sys
import os.path as op


def main():
    sys.path.insert(0, op.abspath('.'))
    from pynets.__about__ import __version__
    print(__version__)


if __name__ == '__main__':
    main()
