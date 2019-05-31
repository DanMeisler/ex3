import numpy as np
import argparse

BLANK_CHARACTER = ' '


def get_input(args):
    y = np.load(args.net_output_path)
    p = list(args.phonemes)
    alphabet = list(args.alphabet) + [BLANK_CHARACTER]
    assert y.shape[0] >= len(p), "output matrix doesnt match phonemes to be classified"
    assert y.shape[1] == len(alphabet), "output matrix doesnt match alphabet to be classified"
    return y, p, alphabet


def parse_args():
    parser = argparse.ArgumentParser("CNC implementation")
    parser.add_argument("net_output_path")
    parser.add_argument("phonemes")
    parser.add_argument("alphabet")
    return parser.parse_args()


def main(args):
    y, p, alphabet = get_input(args)
    z = [BLANK_CHARACTER] + list(' '.join(p)) + [BLANK_CHARACTER]
    print(y)
    print(z)


if __name__ == "__main__":
    main(parse_args())

