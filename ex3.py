import numpy as np
import argparse

BLANK_CHARACTER = ' '


class CTC(object):
    def __init__(self, y, alphabet):
        assert y.shape[1] == len(alphabet) + 1, "output matrix doesnt match alphabet"
        self._y = y
        self._alphabet = alphabet + [BLANK_CHARACTER]
        self._z = None

    def forward(self, p):
        self._z = [BLANK_CHARACTER] + list(' '.join(p)) + [BLANK_CHARACTER]
        return (self._get_alpha(len(self._z) - 1, self._y.shape[0] - 1) +
                self._get_alpha(len(self._z) - 2, self._y.shape[0] - 1))

    def _get_alpha(self, s, t):
        if t == 0:
            if s < 2:
                return self._get_y_probability(1, self._z[s])
            return 0

        if s == 0:
            return self._get_alpha(s, t - 1) * self._get_y_probability(t, self._z[s])

        if (s == 1) or (self._z[s] == BLANK_CHARACTER) or (self._z[s] == self._z[s - 2]):
            return (self._get_alpha(s - 1, t - 1) + self._get_alpha(s, t - 1)) * self._get_y_probability(t, self._z[s])

        return (self._get_alpha(s - 2, t - 1) + self._get_alpha(s - 1, t - 1) +
                self._get_alpha(s, t - 1)) * self._get_y_probability(t, self._z[s])

    def _get_y_probability(self, t, pi):
        assert pi in self._alphabet, "%c not in the alphabet" % pi
        assert t <= self._y.shape[0], "output matrix has no %dth, max is %d" % (t, self._y.shape[0])
        return self._y[t, self._alphabet.index(pi)]


def parse_args():
    parser = argparse.ArgumentParser("CNC implementation")
    parser.add_argument("net_output_path")
    parser.add_argument("phonemes")
    parser.add_argument("alphabet")
    return parser.parse_args()


def main(args):
    ctc = CTC(np.load(args.net_output_path), list(args.alphabet))
    print(ctc.forward(list(args.phonemes)))


if __name__ == "__main__":
    main(parse_args())
