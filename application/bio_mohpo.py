from flaml.model import MyEstimator
from flaml import AutoML, CFO, tune
from collections import defaultdict
import pandas as pd
import argparse
import pickle
import sys
import os
import optuna
import arff
import numpy as np
from ray import tune as raytune
import sys
from flaml.data import load_openml_dataset
import time
import torch
import random


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


data_func = {"brest": 1165, "colon": 1137}

parser = argparse.ArgumentParser()
parser.add_argument("--seed", help="seeds", type=int, nargs="+")
parser.add_argument("--data", help="data", type=str, default="adult")
parser.add_argument("--budget", help="budget", type=int, default=3600)

args = parser.parse_args()
data_name = args.data
seed = args.seed[0]
budget = args.budget
dataset_name = args.data

set_seed(seed)
X_train, X_test, y_train, y_test = load_openml_dataset(dataset_id=data_func[dataset_name], data_dir="./download/")

concat_1 = [X_train,X_test]
X_train  = pd.concat(concat_1)
concat_2 = [y_train, y_test]
y_train = pd.concat(concat_2)

metric = "real_accuracy"
task = "classification"

path = "/workspaces/FLAML/result/bio/mohpo-seed_{seed}_data_{data}/".format(
    seed=seed, data=dataset_name)
if not os.path.isdir(path):
    os.makedirs(path)
logpath = open(os.path.join(path, "log.log"), "w")
sys.stdout = logpath
sys.stderr = logpath


automl = AutoML()
automl.add_learner(learner_name="my_xgboost", learner_class=MyEstimator)
automl._state.eval_only = True
automl._state.fairness_info = None
settings = {
    "time_budget": 0,
    "metric": metric,
    "estimator_list": [
        "my_xgboost"
    ],
    "use_ray": False,
    "task": task,
    "max_iter": 0,
    "keep_search_state": True,
    "log_training_metric": True,
    "verbose": 0,
    "eval_method": "cv",
    "mem_thres": 128 * (1024**3),
    "seed": seed,
}
automl.fit(X_train=X_train, y_train=y_train, **settings)
evaluation_function = automl.trainable

upper = max(5, min(32768, int(X_train.shape[0])))
feature_upper = X_train.shape[1]
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
        "feature_num": trial.suggest_int("feature_num", 1, feature_upper),
        "get_stable": True,
        "learner": "my_xgboost",
    }
    result, live_model = evaluation_function(param)
    trial.set_user_attr("wall_clock_time", time.time() - time_start)
    return 1 - result["val_loss"], result["feature_num"], result["stability"]


Func = optuna.integration.botorch._get_default_candidates_func(n_objectives=2)
sampler = optuna.integration.BoTorchSampler(candidates_func=Func, n_startup_trials=1)

study = optuna.create_study(
    directions=["maximize", "minimize", "minimize"],
    sampler=sampler,
)

study.optimize(objective, timeout=budget)

trials = sorted(study.trials, key=lambda t: t.user_attrs["wall_clock_time"])
histories = defaultdict(list)
for index, trial in enumerate(trials):
    histories["val_loss"].append(1 - trial.values[0])
    histories["wall_clock_time"].append(trial.user_attrs["wall_clock_time"])
    histories["config"].append(trial.params)
    histories["stability"].append(trial.values[2])
    histories["feature_num"].append(trial.values[1])

length = len(histories["val_loss"])
print(histories)
for i in range(length):
    print("---------------------------")
    print("tiral", i)
    print("wall_clock_time", histories["wall_clock_time"][i])
    print("val_loss", histories["val_loss"][i])
    print("feture_number", histories["feature_num"][i])
    print("stability", histories["stability"][i])
    print("config", histories["config"][i])
    print("---------------------------")

savepath = os.path.join(path, "result.pckl")
f = open(savepath, "wb")
pickle.dump(histories, f)
f.close()
logpath.close()

# python bio_mohpo.py --seed 1 --data german --fairmetric DSP --budget 3600
# taskset -c 0,1 nohup python bio_mohpo.py --seed 1 --data german --fairmetric DSP --budget 3600 &
