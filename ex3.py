import numpy as np
import argparse


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
    print(y)


if __name__ == "__main__":
    main(parse_args())
