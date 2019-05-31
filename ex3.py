import numpy as np
import argparse


def assert_input(y, p, alphabet):
    assert y.shape[0] >= len(p), "output matrix doesnt match phonemes to be classified"
    assert y.shape[1] == len(alphabet) + 1, "output matrix doesnt match alphabet to be classified"


def load_output_matrix(net_output_path):
    return np.load(net_output_path)


def parse_args():
    parser = argparse.ArgumentParser("CNC implementation")
    parser.add_argument("net_output_path")
    parser.add_argument("phonemes")
    parser.add_argument("alphabet")
    return parser.parse_args()


def main(args):
    y = load_output_matrix(args.net_output_path)
    p = args.phonemes
    alphabet = args.alphabet
    assert_input(y, p, alphabet)
    print(y)


if __name__ == "__main__":
    main(parse_args())

