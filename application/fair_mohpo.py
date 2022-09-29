import thop
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from flaml import AutoML, CFO, tune
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
from application.data import FairnessDataset

parser = argparse.ArgumentParser()
parser.add_argument("--seed", help="seeds", type=int, nargs="+")
parser.add_argument("--data", help="data", type=str, default="adult")
parser.add_argument("--fairmetric", help="fairmetric", type=str, default="DSP")
parser.add_argument("--budget", help="budget", type=int, default=3600)
args = parser.parse_args()

data_name = args.data
seed = args.seed[0]
budget = args.budget
fairmetric = args.fairmetric

if args.data in ["german","bank"]:
    data_source = 'aif360'
elif args.data in ["adult", "compas"]:
    data_source = 'aif360'
dataset_name = args.data
metric = "accuracy"
task = "classification"

path = "/workspaces/FLAML/result/fairness/MOHPO-seed_{seed}_data_{data}_fairmetric_{fairmetric}/".format(
    seed=seed, data=dataset_name, fairmetric=fairmetric)
if not os.path.isdir(path):
    os.makedirs(path)
logpath = open(os.path.join(path, "log.log"), "w")
sys.stdout = logpath
sys.stderr = logpath

fairness_dataset = FairnessDataset(data_source=data_source, random_state=seed, dataset_name=dataset_name)
X_train, y_train = fairness_dataset.X_train, fairness_dataset.y_train
X_test, y_test = fairness_dataset.X_test, fairness_dataset.y_test

fairness_info = {"fair_metric": fairmetric, "sensitive_attr": fairness_dataset.sensitive_attr}
automl = AutoML()
automl._state.eval_only = True
automl._state.fairness_info = fairness_info
settings = {
    "time_budget": 0,
    "metric": metric,
    "estimator_list": [
        "xgboost"
    ],
    "use_ray": False,
    "task": task,
    "max_iter": 0,
    "train_time_limit": 60,
    "keep_search_state": True,
    "log_training_metric": True,
    "verbose": 0,
    "eval_method": "holdout",
    "mem_thres": 128 * (1024**3),
    "seed": seed,
}
automl.fit(X_train=X_train, y_train=y_train, X_val=X_test, y_val=y_test, **settings)
evaluation_function = automl.trainable

upper = max(5, min(32768, int(X_train.shape[0])))
time_start = time.time()
def objective(trial):
    param = {
        'max_leaves': trial.suggest_int("n_estimators", 4, upper, log=True),
        'max_depth': trial.suggest_categorical('max_depth', [0, 6, 12]),
        "n_estimators": trial.suggest_int("n_estimators", 4, upper, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.001, 128, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 1 / 1024, 1.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.1, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.01, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1 / 1024, 1024, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1 / 1024, 1024, log=True),
        "learner": "xgboost",
    }
    result, live_model = evaluation_function(param)
    trial.set_user_attr("wall_clock_time", time.time() - time_start)
    return 1 - result["val_loss"], result["fairness"]

Func = optuna.integration.botorch._get_default_candidates_func(n_objectives=2)
sampler = optuna.integration.BoTorchSampler(candidates_func=Func, n_startup_trials=1)

study = optuna.create_study(directions=["maximize", "minimize"], sampler=sampler)
study.optimize(objective, timeout=budget)

trials = sorted(study.trials, key=lambda t: t.user_attrs["wall_clock_time"])
histories = defaultdict(list)
for index, trial in enumerate(trials):
    histories["val_loss"].append(1 - trial.values[0])
    histories["wall_clock_time"].append(trial.user_attrs["wall_clock_time"])
    histories["config"].append(trial.params)
    histories["fairness"].append(trial.values[1])

length = len(histories["val_loss"])
for i in range(length):
    print("---------------------------")
    print("tiral", i)
    print("config", histories["config"][i])
    print("val_loss", histories["val_loss"][i])
    print("fairness", histories["fairness"][i])
    print("---------------------------")

savepath = os.path.join(path, "result.pckl")
f = open(savepath, "wb")
pickle.dump(histories, f)
f.close()
logpath.close()

# python fair_mohpo.py --seed 1 --data german --fairmetric DSP --budget 3600
# taskset -c 0,1 nohup python fair_mohpo.py --seed 1 --data german --fairmetric DSP --budget 3600 &
