import torch
import numpy as np


class Discretizer():
    def __init__(self, ranges, num_bins) -> None:
        self.ranges = ranges
        self.num_bins = num_bins
        self.boundaries = torch.tensor(np.array([
            np.linspace(x, y, num_bins) for x, y in self.ranges
        ]))

    def __call__(self, x) -> torch.Any:
        r = torch.zeros(x.shape)
        for i in range(x.shape[-1]):
            distances = torch.stack(
                [torch.abs(self.boundaries[i] - v).argmin() for v in x[..., i]])
            r[..., i] = distances
        return r.to(torch.int)

    def reverse(self, x):
        r = torch.zeros(x.shape)
        for i in range(x.shape[-1]):
            r[..., i] = torch.tensor(
                [self.boundaries[i][int(b)] for b in x[..., i]])
        return r

    def one_hot(self, x):
        x = x.long()
        return torch.nn.functional.one_hot(x, num_classes=self.num_bins)  # pylint: disable=not-callable


def test_discretizer():
    action_ranges = [
        (-1, 1),
        (0, 1)
    ]
    discretizer = Discretizer(
        ranges=action_ranges,
        num_bins=3
    )
    print(f"Boundaries: {discretizer.boundaries}")
    dummy_action = torch.tensor([[-0.2, 0.8], [1.1, 0.7]])
    print(f"Action: {dummy_action}")
    discrete = discretizer(dummy_action)
    print(f"Discrete: {discrete}")
    reversed_action = discretizer.reverse(discrete)
    print(f"Reversed: {reversed_action}")
    one_hot = discretizer.one_hot(discrete)
    print(f"One-hot: {one_hot}")


if __name__ == "__main__":
    test_discretizer()
