import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from model.manicast import ManiCast
from utils.ang2joint import *
from utils.loss_funcs import (
    mpjpe_error,
    fde_error,
    weighted_mpjpe_error,
    perjoint_error,
)
from utils.parser import args
from data.utils.costs import *
from data.utils.transitions import *
from utils.cost_funcs import calc_reactive_stirring_task_cost
from utils.read_json_data import read_json
from torch.utils.tensorboard import SummaryWriter
import pathlib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device: %s" % device)


def update_step(
    epoch,
    model,
    writer,
    joint_used,
    joint_names,
    joint_weights,
    dataloader,
    optimizer=None,
    eval=True,
    step_type="test",
):
    if eval:
        model.eval()
    else:
        model.train()
    running_loss = 0
    running_all_joints_error = 0
    running_per_joint_error = 0
    running_fde = 0
    running_cost_forecast = 0
    running_cost_future = 0
    running_cost_dif = 0
    n = 0
    for cnt, batch in enumerate(dataloader):
        forecast_human, other_human, is_reaching = [
            b.float().to(device) for b in batch
        ]  # multiply by 1000 for milimeters
        forecast_human_history = forecast_human[:, 0 : args.input_n, :, :].permute(
            0, 3, 1, 2
        )
        forecast_human_left_hip = forecast_human_history[
            :, :, args.input_n - 1 :, 21:22
        ]
        forecast_human_history_offset = (
            forecast_human_history[:, :, :, joint_used] - forecast_human_left_hip
        )
        forecast_human_predict_gt = forecast_human[
            :, args.input_n : args.input_n + args.output_n, joint_used, :
        ]

        if not eval:
            optimizer.zero_grad()
        forecast_human_predict_offset = model(forecast_human_history_offset)
        forecast_human_predict = forecast_human_predict_offset.permute(
            0, 1, 3, 2
        ) + forecast_human_left_hip.permute(0, 2, 3, 1)
        other_human_future = other_human[:, :, joint_used, :]

        forecast_cost = calc_reactive_stirring_task_cost(
            forecast_human_predict, other_human_future, is_reaching
        )
        future_cost = calc_reactive_stirring_task_cost(
            forecast_human_predict_gt, other_human_future, is_reaching
        )
        cost_dif = torch.sum(torch.abs(forecast_cost - future_cost), dim=1)
        all_joints_error = (
            weighted_mpjpe_error(
                forecast_human_predict, forecast_human_predict_gt, joint_weights
            )
            * 1000
        )

        if args.weight_using == "forecast":
            loss = torch.mean(
                torch.exp(args.cost_weight * (torch.sum(forecast_cost, dim=1)))
                * all_joints_error
            )
        elif args.weight_using == "future":
            loss = torch.mean(
                torch.exp(args.cost_weight * (torch.sum(future_cost, dim=1)))
                * all_joints_error
            )
        elif args.weight_using == "difference":
            loss = torch.mean(torch.exp(args.cost_weight * cost_dif) * all_joints_error)

        per_joint_error = (
            perjoint_error(forecast_human_predict, forecast_human_predict_gt) * 1000
        )
        fde = fde_error(forecast_human_predict, forecast_human_predict_gt) * 1000
        if not eval:
            loss.backward()
            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

        #### Save all metrics for logging
        batch_dim = forecast_human.shape[0]
        n += batch_dim
        running_loss += loss * batch_dim
        running_per_joint_error += per_joint_error * batch_dim
        running_all_joints_error += torch.mean(all_joints_error) * batch_dim
        running_fde += fde * batch_dim
        running_cost_forecast += torch.mean(torch.sum(forecast_cost, dim=1)) * batch_dim
        running_cost_future += torch.mean(torch.sum(future_cost, dim=1)) * batch_dim
        running_cost_dif += torch.mean(cost_dif) * batch_dim

    print(f"[{epoch+1}]  {step_type} loss: {round(running_loss.item()/n, 3)}")
    writer.add_scalar(f"{step_type}/mpjpe", running_all_joints_error.item() / n, epoch)
    writer.add_scalar(f"{step_type}/loss", running_loss.item() / n, epoch)
    writer.add_scalar(f"{step_type}/fde", running_fde.item() / n, epoch)
    writer.add_scalar(
        f"{step_type}/forecast_cost", running_cost_forecast.item() / n, epoch
    )
    writer.add_scalar(f"{step_type}/future_cost", running_cost_future.item() / n, epoch)
    writer.add_scalar(f"{step_type}/cost_dif", running_cost_dif.item() / n, epoch)
    for idx, joint in enumerate(["LWristOut", "RWristOut"]):
        writer.add_scalar(
            f"{step_type}/{joint}_error",
            running_per_joint_error[idx + 5].item() / n,
            epoch,
        )


def train(model, writer, joint_used, joint_names, model_name, joint_weights):
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-05)

    Dataset = CostDataset(
        "./data/comad_data", args.input_n, args.output_n, sample_rate=25, split=0
    )
    Dataset_val = CostDataset(
        "./data/comad_data", args.input_n, args.output_n, sample_rate=25, split=1
    )
    Dataset_test = CostDataset(
        "./data/comad_data", args.input_n, args.output_n, sample_rate=25, split=2
    )

    loader_train = DataLoader(
        Dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )

    loader_val = DataLoader(
        Dataset_val, batch_size=args.batch_size, shuffle=True, num_workers=0
    )

    loader_test = DataLoader(
        Dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    model.train()
    for epoch in range(args.n_epochs):
        update_step(
            epoch,
            model,
            writer,
            joint_used,
            joint_names,
            joint_weights,
            loader_train,
            optimizer=optimizer,
            eval=False,
            step_type="train",
        )
        with torch.no_grad():
            # Validation step
            update_step(
                epoch,
                model,
                writer,
                joint_used,
                joint_names,
                joint_weights,
                loader_val,
                optimizer=None,
                eval=True,
                step_type="val",
            )

            # Test step
            update_step(
                epoch,
                model,
                writer,
                joint_used,
                joint_names,
                joint_weights,
                loader_test,
                optimizer=None,
                eval=True,
                step_type="test",
            )

        print("----saving model-----")

        pathlib.Path("./model_checkpoints/" + args.model_path + "_ft_car").mkdir(
            parents=True, exist_ok=True
        )
        torch.save(
            model.state_dict(),
            os.path.join(f"./model_checkpoints/{args.model_path}_ft_car", f"{epoch}"),
        )


if __name__ == "__main__":
    weight = args.weight
    joint_weights_base = torch.tensor([1, 1, 1, 1, 1, 1, 1]).float().to(device)
    joint_weights = joint_weights_base.unsqueeze(0).unsqueeze(0).unsqueeze(3)
    joint_names = [
        "BackTop",
        "LShoulderBack",
        "RShoulderBack",
        "LElbowOut",
        "RElbowOut",
        "LWristOut",
        "RWristOut",
    ]
    mapping = read_json("./data/mapping.json")
    joint_used = np.array([mapping[joint_name] for joint_name in joint_names])
    model = ManiCast(
        args.input_dim,
        args.input_n,
        args.output_n,
        args.st_gcnn_dropout,
        args.joints_to_consider,
        args.n_tcnn_layers,
        args.tcnn_kernel_size,
        args.tcnn_dropout,
    ).to(device)
    model_name = "amass_3d_" + str(args.output_n) + "frames_ckpt"
    model.load_state_dict(torch.load(f"./model_checkpoints/{args.load_path}"))
    print(
        "total number of parameters of the network is: "
        + str(sum(p.numel() for p in model.parameters() if p.requires_grad))
    )
    writer = SummaryWriter(log_dir="./finetune_logs_cost_weight/" + args.model_path)
    train(model, writer, joint_used, joint_names, model_name, joint_weights)