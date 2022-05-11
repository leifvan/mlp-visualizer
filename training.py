from typing import Tuple, List, Type
import torch
import numpy as np
from sklearn import datasets


class CirclesDataset:

    def __init__(self, n_samples: int = 10000) -> None:
        self.centers, self.labels = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)

    def __len__(self) -> int:
        return len(self.centers)

    def __getitem__(self, idx: int) -> Tuple[np.float32, np.float32]:
        return self.centers[idx], self.labels[idx].astype(np.float32)


class BinaryClassifierMLP(torch.nn.Module):

    def __init__(self, n_inputs: int, layers: List, activation_fn: Type) -> None:
        super().__init__()

        self.layers = torch.nn.ModuleList()
        previous_num_neurons = n_inputs

        for num_neurons in layers:
            layer = torch.nn.Linear(previous_num_neurons, num_neurons)
            self.layers.append(layer)
            self.layers.append(activation_fn())
            previous_num_neurons = num_neurons

        self.layers.append(torch.nn.Linear(previous_num_neurons, 1))
        self.layers.append(torch.nn.Sigmoid())

    @torch.no_grad()
    def update_model(self, w1, w2, wo, b1, b2, bo, act):
        self.layers[0].weight[:] = w1
        self.layers[0].bias[:] = b1
        self.layers[1] = act
        self.layers[2].weight[:] = w2
        self.layers[2].bias[:] = b2
        self.layers[3] = act
        self.layers[4].weight[:] = wo
        self.layers[4].bias[:] = bo

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x.squeeze()

    def forward_until(self, x: torch.Tensor, layer: int) -> torch.Tensor:
        for layer in self.layers[:layer+1]:
            x = layer(x)
        return x.squeeze()


def train_binary(
        epochs: int,
        dataloader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        optim: torch.optim.Optimizer
) -> List:
    loss_curve = []
    for epoch in range(epochs):
        epoch_losses = []
        for batch_idx, (input_data, gt_label) in enumerate(dataloader):
            optim.zero_grad()
            predicted = model(input_data.to(torch.float32))
            loss = torch.nn.functional.binary_cross_entropy(predicted, gt_label)
            loss.backward()
            optim.step()
            epoch_losses.append(loss.item())
        loss_curve.append(sum(epoch_losses) / len(epoch_losses))
    return loss_curve


@torch.no_grad()
def evaluate(
        dataloader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
) -> float:
    num_correct = 0
    num_total = 0
    for step, (input_data, gt_label) in enumerate(dataloader):
        predicted = torch.round(model(input_data.to(torch.float32)))
        num_correct += (predicted == gt_label).count_nonzero()
        num_total += len(input_data)
    return num_correct / num_total
