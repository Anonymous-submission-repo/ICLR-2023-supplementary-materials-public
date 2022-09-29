import thop
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import torchvision
import argparse
import optuna
import random
import numpy as np
import sys
import time
import os
import pickle
from optuna.multi_objective.samplers import RandomMultiObjectiveSampler

DEVICE = torch.device("cpu")
DIR = "./download/"
BATCHSIZE = 128
N_TRAIN_EXAMPLES = BATCHSIZE * 30
N_VALID_EXAMPLES = BATCHSIZE * 10


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


parser = argparse.ArgumentParser()
parser.add_argument("--seed", help="seeds", type=int, nargs="+")
parser.add_argument("--budget", help="budget", type=int, default=3600)
parser.add_argument("--second_obj", help="second_obj", type=str, default="flops")

args = parser.parse_args()
seed = args.seed[0]
budget = args.budget
second_obj = args.second_obj

path = "/workspaces/FLAML/result/NN/mohpo-seed_{seed}_obj_{obj}/".format(seed=seed, obj=second_obj)
if not os.path.isdir(path):
    os.makedirs(path)
logpath = open(os.path.join(path, "log.log"), "w")
sys.stdout = logpath
sys.stderr = logpath

train_dataset = torchvision.datasets.FashionMNIST(
    DIR, train=True, download=True, transform=torchvision.transforms.ToTensor()
)
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.Subset(train_dataset, list(range(N_TRAIN_EXAMPLES))),
    batch_size=BATCHSIZE,
    shuffle=True,
)

val_dataset = torchvision.datasets.FashionMNIST(
    DIR, train=False, transform=torchvision.transforms.ToTensor()
)
val_loader = torch.utils.data.DataLoader(
    torch.utils.data.Subset(val_dataset, list(range(N_VALID_EXAMPLES))),
    batch_size=BATCHSIZE,
    shuffle=True,
)


def define_model(trial):
    n_layers = trial.suggest_int("n_layers", 1, 3)
    layers = []
    in_features = 28 * 28
    for i in range(n_layers):
        out_features = trial.suggest_int("n_units_l{}".format(i), 4, 128)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        p = trial.suggest_float("dropout_{}".format(i), 0.2, 0.5)
        layers.append(nn.Dropout(p))
        in_features = out_features
    layers.append(nn.Linear(in_features, 10))
    layers.append(nn.LogSoftmax(dim=1))
    return nn.Sequential(*layers)


def train_model(model, optimizer, train_loader):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.view(-1, 28 * 28).to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        F.nll_loss(model(data), target).backward()
        optimizer.step()


def eval_model(model, valid_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(valid_loader):
            data, target = data.view(-1, 28 * 28).to(DEVICE), target.to(DEVICE)
            pred = model(data).argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = correct / N_VALID_EXAMPLES
    flops, params = thop.profile(model, inputs=(torch.randn(1, 28 * 28).to(DEVICE),), verbose=False)
    return np.log2(flops), accuracy, params


time_start_flops = time.time()
def objective_flops(trial):
    trial.set_user_attr("wall_clock_time", time.time() - time_start_flops)
    model = define_model(trial).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(), trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    )
    n_epoch = trial.suggest_int("n_epoch", 1, 20)
    train_start = time.time()
    for epoch in range(n_epoch):
        train_model(model, optimizer, train_loader)
    training_time = time.time() - train_start
    flops, accuracy, params = eval_model(model, val_loader)
    trial.set_user_attr("training_time", training_time)
    trial.set_user_attr("params", params)
    return accuracy, flops

time_start_training_time = time.time()
def objective_training_time(trial):
    trial.set_user_attr("wall_clock_time", time.time() - time_start_training_time)
    model = define_model(trial).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(), trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    )
    n_epoch = trial.suggest_int("n_epoch", 1, 20)
    train_start = time.time()
    for epoch in range(n_epoch):
        train_model(model, optimizer, train_loader)
    training_time = time.time() - train_start
    flops, accuracy, params = eval_model(model, val_loader)
    trial.set_user_attr("flops", flops)
    trial.set_user_attr("params", params)
    return accuracy, training_time

time_start_params = time.time()
def objective_params(trial):
    trial.set_user_attr("wall_clock_time", time.time() - time_start_params)
    model = define_model(trial).to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(), trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    )
    n_epoch = trial.suggest_int("n_epoch", 1, 20)
    train_start = time.time()
    for epoch in range(n_epoch):
        train_model(model, optimizer, train_loader)
    flops, accuracy, params = eval_model(model, val_loader)
    training_time = time.time() - train_start
    trial.set_user_attr("flops", flops)
    trial.set_user_attr("training_time", training_time)
    return accuracy, params



Func = optuna.integration.botorch._get_default_candidates_func(n_objectives=2)
sampler = optuna.integration.BoTorchSampler(candidates_func=Func, n_startup_trials=1)

study = optuna.create_study(directions=["maximize", "minimize"], sampler=sampler)
if second_obj == "training_time":
    study.optimize(objective_training_time, timeout=budget)
elif second_obj == "flops":
    study.optimize(objective_flops, timeout=budget)
else:
    study.optimize(objective_params, timeout=budget) 

trials = sorted(study.trials, key=lambda t: t.user_attrs["wall_clock_time"])
histories = defaultdict(list)
for index, trial in enumerate(trials):
    histories["error_rate"].append(1 - trial.values[0])
    histories["wall_clock_time"].append(trial.user_attrs["wall_clock_time"])
    histories["config"].append(trial.params)
    if second_obj == "training_time":
        histories["training_time"].append(trial.values[1])
        histories["flops"].append(trial.user_attrs["flops"])
        histories["params"].append(trial.user_attrs["params"])
    elif second_obj == "flops":
        histories["flops"].append(trial.values[1])
        histories["training_time"].append(trial.user_attrs["training_time"])
        histories["params"].append(trial.user_attrs["params"])
    else:
        histories["params"].append(trial.values[1])
        histories["training_time"].append(trial.user_attrs["training_time"])
        histories["flops"].append(trial.user_attrs["flops"])

length = len(histories["error_rate"])
for i in range(length):
    print("---------------------------")
    print("tiral", i)
    print("config", histories["config"][i])
    print("error_rate", histories["error_rate"][i])
    print("flops", histories["flops"][i])
    print("training_time", histories["training_time"][i])
    print("params", histories["params"][i])
    print("---------------------------")

savepath = os.path.join(path, "result.pckl")
f = open(savepath, "wb")
pickle.dump(histories, f)
f.close()
logpath.close()

# python nn_mohpo.py --seed 1 --budget 3600 --second_obj training_time
