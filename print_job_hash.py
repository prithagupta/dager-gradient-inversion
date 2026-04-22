#!/usr/bin/env python
import sys

from args_factory import get_args
from utils.experiment import get_hash_value_for_args


def main(argv=None):
    args = get_args(argv)
    print(get_hash_value_for_args(args))


if __name__ == "__main__":
    main(sys.argv[1:])
