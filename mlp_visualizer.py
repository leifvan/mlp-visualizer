from io import BytesIO
from random import randint

import PySimpleGUI as sg
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from training import CirclesDataset, BinaryClassifierMLP, train_binary, evaluate

scale = 100


def make_slider(key, alt=False):
    return sg.Slider(
        range=(-10 * scale, 10 * scale),
        default_value=randint(-10 * scale, 10 * scale),
        orientation='h',
        size=(15, 20),
        key=key,
        background_color='#CC5555' if alt else None
    )


def params_to_arrays(p):
    w1 = torch.tensor([[p["w111"], p["w112"]],
                       [p["w121"], p["w122"]],
                       [p["w131"], p["w132"]],
                       [p["w141"], p["w142"]]]) / scale
    b1 = torch.tensor([p["b11"], p["b12"], p["b13"], p["b14"]]) / scale
    w2 = torch.tensor([[p["w211"], p["w212"], p["w213"], p["w214"]],
                       [p["w221"], p["w222"], p["w223"], p["w224"]],
                       [p["w231"], p["w232"], p["w233"], p["w234"]],
                       [p["w241"], p["w242"], p["w243"], p["w244"]]]) / scale
    b2 = torch.tensor([p["b21"], p["b22"], p["b23"], p["b24"]]) / scale

    wo = torch.tensor([[p["wo1"], p["wo2"], p["wo3"], p["wo4"]]]) / scale
    bo = torch.tensor([p["bo"]]) / scale

    return w1, w2, wo, b1, b2, bo


def get_activation_function(params):
    if params["ReLU"]:
        return torch.nn.ReLU()
    elif params["Tanh"]:
        return torch.nn.Tanh()
    elif params["Sigmoid"]:
        return torch.nn.Sigmoid()
    return torch.nn.Identity()


def compute_grid(model):
    xx, yy = torch.meshgrid(torch.linspace(-2, 2, 400), torch.linspace(-2, 2, 400))
    points = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    return model(points).reshape(400, 400)


def update_parameters_window(window, model):
    # update screen components
    window.Element("w111").Update(int(scale * model.layers[0].weight[0, 0]))
    window.Element("w112").Update(int(scale * model.layers[0].weight[0, 1]))
    window.Element("w121").Update(int(scale * model.layers[0].weight[1, 0]))
    window.Element("w122").Update(int(scale * model.layers[0].weight[1, 1]))
    window.Element("w131").Update(int(scale * model.layers[0].weight[2, 0]))
    window.Element("w132").Update(int(scale * model.layers[0].weight[2, 1]))
    window.Element("w141").Update(int(scale * model.layers[0].weight[3, 0]))
    window.Element("w142").Update(int(scale * model.layers[0].weight[3, 1]))
    window.Element("b11").Update(int(scale * model.layers[0].bias[0]))
    window.Element("b12").Update(int(scale * model.layers[0].bias[1]))

    window.Element("w211").Update(int(scale * model.layers[2].weight[0, 0]))
    window.Element("w212").Update(int(scale * model.layers[2].weight[0, 1]))
    window.Element("w213").Update(int(scale * model.layers[2].weight[0, 2]))
    window.Element("w214").Update(int(scale * model.layers[2].weight[0, 3]))
    window.Element("w221").Update(int(scale * model.layers[2].weight[1, 0]))
    window.Element("w222").Update(int(scale * model.layers[2].weight[1, 1]))
    window.Element("w223").Update(int(scale * model.layers[2].weight[1, 2]))
    window.Element("w224").Update(int(scale * model.layers[2].weight[1, 3]))
    window.Element("w231").Update(int(scale * model.layers[2].weight[2, 0]))
    window.Element("w232").Update(int(scale * model.layers[2].weight[2, 1]))
    window.Element("w233").Update(int(scale * model.layers[2].weight[2, 2]))
    window.Element("w234").Update(int(scale * model.layers[2].weight[2, 3]))
    window.Element("w241").Update(int(scale * model.layers[2].weight[3, 0]))
    window.Element("w242").Update(int(scale * model.layers[2].weight[3, 1]))
    window.Element("w243").Update(int(scale * model.layers[2].weight[3, 2]))
    window.Element("w244").Update(int(scale * model.layers[2].weight[3, 3]))
    window.Element("b21").Update(int(scale * model.layers[2].bias[0]))
    window.Element("b22").Update(int(scale * model.layers[2].bias[1]))
    window.Element("b23").Update(int(scale * model.layers[2].bias[2]))
    window.Element("b24").Update(int(scale * model.layers[2].bias[3]))

    window.Element("wo1").Update(int(scale * model.layers[4].weight[0, 0]))
    window.Element("wo2").Update(int(scale * model.layers[4].weight[0, 1]))
    window.Element("wo3").Update(int(scale * model.layers[4].weight[0, 2]))
    window.Element("wo4").Update(int(scale * model.layers[4].weight[0, 3]))
    window.Element("bo").Update(int(scale * model.layers[4].bias[0]))


def main():
    dataset = CirclesDataset(n_samples=10_000)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    centers_0 = torch.round(torch.from_numpy(200 + 100 * dataset.centers[dataset.labels == 0])).to(torch.long)
    centers_1 = torch.round(torch.from_numpy(200 + 100 * dataset.centers[dataset.labels == 1])).to(torch.long)

    model = BinaryClassifierMLP(n_inputs=2, layers=[4, 4], activation_fn=torch.nn.Identity)
    optimizer = torch.optim.Adam(model.parameters())

    layout = [[
        [sg.Text("Layer 1")],
        [make_slider(f"w11{i + 1}") for i in range(2)],
        [make_slider(f"w12{i + 1}") for i in range(2)],
        [make_slider(f"w13{i + 1}") for i in range(2)],
        [make_slider(f"w14{i + 1}") for i in range(2)],
        [make_slider(f"b1{i + 1}", alt=True) for i in range(4)],
        [sg.Text("Layer 2")],
        [make_slider(f"w21{i + 1}") for i in range(4)],
        [make_slider(f"w22{i + 1}") for i in range(4)],
        [make_slider(f"w23{i + 1}") for i in range(4)],
        [make_slider(f"w24{i + 1}") for i in range(4)],
        [make_slider(f"b2{i + 1}", alt=True) for i in range(4)],
        [sg.Text("Output")],
        *[[make_slider(f"wo{i + 1}")] for i in range(4)],
        [make_slider(f"bo", alt=True)],
        [sg.Radio("Identity", group_id="activation", key="Identity", default=True)]
        + [sg.Radio(v, group_id="activation", key=v) for v in ("ReLU", "Tanh", "Sigmoid")],
        [
            sg.Button("Optimize (1 epoch)"),
            sg.Button("Optimize (10 epochs)"),
            sg.Button("Optimize (100 epochs)"),
            sg.Button("Reset")
        ]
    ]]

    vis_image = sg.Image()
    vis_accuracy = sg.Text()
    parameters_window = sg.Window("Parameters", layout)
    visualization_window = sg.Window("Visualization", [[vis_image], [vis_accuracy]])

    last_values = {}
    num_optimize_steps = 0

    while True:
        event, values = visualization_window.read(timeout=0.2)
        if event == sg.WIN_CLOSED or event == 'Cancel':  # if user closes window or clicks cancel
            break

        event, values = parameters_window.read(timeout=0.2)
        if event == sg.WIN_CLOSED or event == 'Cancel':  # if user closes window or clicks cancel
            break
        if last_values != values:
            last_values = values
            arrays = params_to_arrays(values)
            act_fn = get_activation_function(values)

            model.update_model(*arrays, act=act_fn)
            samples = compute_grid(model)
            samples = torch.clip(samples, 0, 1)

            accuracy = evaluate(dataloader, model)
            vis_accuracy.update(f"accuracy = {accuracy:.2%}")

            image = (plt.cm.coolwarm(samples.detach())[..., :3] * 255).astype(np.uint8)
            image[centers_0[:, 0], centers_0[:, 1]] = torch.tensor([0, 0, 255]).to(torch.uint8)
            image[centers_1[:, 0], centers_1[:, 1]] = torch.tensor([255, 0, 0]).to(torch.uint8)

            im = Image.fromarray(image)
            with BytesIO() as output:
                im.save(output, format="PNG")
                data = output.getvalue()

            vis_image.update(data=data)
        elif event == "Optimize (1 epoch)":
            num_optimize_steps += 1
        elif event == "Optimize (10 epochs)":
            num_optimize_steps += 10
        elif event == "Optimize (100 epochs)":
            num_optimize_steps += 100
        elif event == "Reset":
            model = BinaryClassifierMLP(
                n_inputs=2,
                layers=[4, 4],
                activation_fn=lambda: get_activation_function(values)
            )
            optimizer = torch.optim.Adam(model.parameters())
            update_parameters_window(parameters_window, model)

        if num_optimize_steps > 0:
            num_optimize_steps -= 1
            train_binary(1, dataloader, model, optimizer)
            update_parameters_window(parameters_window, model)

            accuracy = evaluate(dataloader, model)
            print(f"optimizing ({num_optimize_steps} more steps), accuracy = {accuracy:.2%}")
            vis_accuracy.update(f"optimizing ({num_optimize_steps} more steps), accuracy = {accuracy:.2%}")

    parameters_window.close()


if __name__ == "__main__":
    main()
