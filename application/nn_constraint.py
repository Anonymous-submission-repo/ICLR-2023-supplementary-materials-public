from distutils.command.config import config
import torch
import thop
import torch.nn as nn
from collections import defaultdict
from flaml import AutoML, CFO, tune
import torch.nn.functional as F
import torchvision
import argparse
import optuna
import random
import numpy as np
import time
import sys
import os
import pickle
from ray import tune as raytune
from optuna.multi_objective.samplers import RandomMultiObjectiveSampler

DEVICE = torch.device("cpu")
DIR = "./download/"
BATCHSIZE = 128
N_TRAIN_EXAMPLES = BATCHSIZE * 30
N_VALID_EXAMPLES = BATCHSIZE * 10


def combine_value(obj1, obj2, tolerance):
    obj1 = int(obj1 // tolerance)
    result = obj1 * 100 + obj2
    return result

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument("--seed", help="seeds", type=int, nargs="+")
parser.add_argument("--budget", help="budget", type=int, default=3600)
parser.add_argument("--tolerance", help="tolerance", type=float, default=0.028)
parser.add_argument("--method", help="method", type=str, default="CFO")
parser.add_argument("--second_obj", help="second_obj", type=str, default="flops")
args = parser.parse_args()
seed = args.seed[0]
budget = args.budget
method = args.method
tolerance = args.tolerance
second_obj = args.second_obj

set_seed(seed)

if method == "Lexico":
    path = "/workspaces/FLAML/result/NN/{method}-seed_{seed}_tolerance_{tolerance}_obj_{obj}/".format(
        method=method, seed=seed, tolerance=tolerance, obj=second_obj)
else:
    path = "/workspaces/FLAML/result/NN/{method}-seed_{seed}_obj_{obj}/".format(
        method=method, seed=seed, tolerance=tolerance, obj=second_obj)

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


def define_model(configuration):
    n_layers = configuration["n_layers"]
    layers = []
    in_features = 28 * 28
    for i in range(n_layers):
        out_features = configuration["n_units_l{}".format(i)]
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        p = configuration["dropout_{}".format(i)]
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
    q_error_rate = combine_value((1 - accuracy), np.log2(flops), tolerance)
    return np.log2(flops), (1 - accuracy), params, q_error_rate


time_start = time.time()


def evaluate_function(configuration):
    model = define_model(configuration).to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(), configuration["lr"]
    )
    n_epoch = configuration["n_epoch"]
    train_start = time.time()
    for epoch in range(n_epoch):
        train_model(model, optimizer, train_loader)
    flops, error_rate, params, q_error_rate = eval_model(model, val_loader)
    return {"error_rate": error_rate, "flops": flops, "wall_clock_time": time.time() - time_start, "training_time": time.time() - train_start, "params": params, "q_error_rate": q_error_rate}


if method == "constraint":
    lexico_info = None
else:
    lexico_info = {}
    lexico_info["metric_priority"] = ["error_rate", second_obj]
    lexico_info["tolerance"] = {"error_rate": tolerance, second_obj: 0.0}
    lexico_info["target"] = {"error_rate": 0.0, second_obj: 0.0}

search_space = {
    "n_layers": raytune.randint(lower=1, upper=3),
    "n_units_l0": raytune.randint(lower=4, upper=128),
    "n_units_l1": raytune.randint(lower=4, upper=128),
    "n_units_l2": raytune.randint(lower=4, upper=128),
    "dropout_0": raytune.uniform(lower=0.2, upper=0.5),
    "dropout_1": raytune.uniform(lower=0.2, upper=0.5),
    "dropout_2": raytune.uniform(lower=0.2, upper=0.5),
    "lr": raytune.loguniform(lower=1e-5, upper=1e-1),
    "n_epoch": raytune.randint(lower=1, upper=20),
}

low_cost_partial_config = {
    "n_layers": 1,
    "n_units_l0": 4,
    "n_units_l1": 4,
    "n_units_l2": 4,
    "n_epoch": 1,
}

# ---------------------------- error rate ------------------------------------
algo = CFO(
    lexico_info=lexico_info,
    space=search_space,
    metric="error_rate",
    mode="min",
    seed=seed,
    low_cost_partial_config=low_cost_partial_config,
)
analysis = tune.run(
    evaluate_function,
    without_live_model=False,
    local_dir=path,
    num_samples=100000000,
    time_budget_s=budget / 2,
    search_alg=algo,
    metric = "error_rate",
    log_file_name="log_flaml",
)
best_result = analysis.best_result["error_rate"]
analysis = analysis.results
keys = list(analysis.keys())
length = len(keys)
objectives = ["error_rate", "flops", "wall_clock_time", "config", "training_time", "params", "q_error_rate"]
histories = defaultdict(list)
for time_index in range(length):
    for objective in objectives:
        histories[objective].append(analysis[keys[time_index]][objective])
# print(len(histories["error_rate"]))
# ---------------------------- error rate ------------------------------------
# ---------------------------- obj2 ------------------------------------
algo = CFO(
    lexico_info=lexico_info,
    space=search_space,
    metric=second_obj,
    mode="min",
    seed=seed,
    low_cost_partial_config=low_cost_partial_config,
)
metrics_constraint = [("error_rate", "<=", best_result + tolerance)]
analysis = tune.run(
    evaluate_function,
    without_live_model=False,
    local_dir=path,
    num_samples=100000000,
    time_budget_s = budget / 2,
    search_alg=algo,
    metric=second_obj,
    use_ray=False,
    metric_constraints=metrics_constraint,
    log_file_name="log_flaml",
)
analysis = analysis.results
keys = list(analysis.keys())
length = len(keys)
objectives = ["error_rate", "flops", "wall_clock_time", "config", "training_time", "params", "q_error_rate"]
for time_index in range(length):
    for objective in objectives:
        histories[objective].append(analysis[keys[time_index]][objective])
# print(len(histories["error_rate"]))
length = len(histories["error_rate"])
# ---------------------------- obj2 ------------------------------------
for i in range(length):
    print("---------------------------")
    print("tiral", i)
    print("q_error_rate", histories["q_error_rate"][i])
    print("error_rate", histories["error_rate"][i])
    print("params", histories["params"][i])
    print("training_time", histories["training_time"][i])
    print("flops", histories["flops"][i])
    print("wall_clock_time", histories["wall_clock_time"][i])
    print("config", histories["config"][i])
    print("---------------------------")
savepath = os.path.join(path, "result.pckl")
f = open(savepath, "wb")
pickle.dump(histories, f)
f.close()
logpath.close()

# python nn_constraint.py --method constraint --seed 1 --budget 50 --tolerance 0.05
