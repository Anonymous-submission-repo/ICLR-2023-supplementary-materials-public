from flaml import AutoML, CFO, tune
from application.data import FairnessDataset
from collections import defaultdict
import argparse
import pickle
import os
import math
from ray import tune as raytune
import sys

low_cost_partial_config = {
    "n_estimators": 4,
    "max_leaves": 4,
}

points_to_evaluate = [{"n_estimators": 4,
                       "max_leaves": 4,
                       "max_depth": 0,
                       "min_child_weight": 1.0,
                       "learning_rate": 0.1,
                       "subsample": 1.0,
                       "colsample_bylevel": 1.0,
                       "colsample_bytree": 1.0,
                       "reg_alpha": 1 / 1024,
                       "reg_lambda": 1.0,
                       }]

parser = argparse.ArgumentParser()
parser.add_argument("--seed", help="seeds", type=int, nargs="+")
parser.add_argument("--data", help="data", type=str, default="adult")
parser.add_argument("--fairmetric", help="fairmetric", type=str, default="DSP")
parser.add_argument("--budget", help="budget", type=int, default=3600)
parser.add_argument("--method", help="method", type=str, default="CFO")
args = parser.parse_args()

data_name = args.data
seed = args.seed[0]
budget = args.budget
fairmetric = args.fairmetric
method = args.method

if args.data in ["german", "bank"]:
    data_source = 'aif360'
elif args.data in ["adult", "compas"]:
    data_source = 'aif360'
dataset_name = args.data
metric = "accuracy"
task = "classification"

path = "/workspaces/FLAML/result/fairness/{method}-seed_{seed}_data_{data}_fairmetric_{fairmetric}/".format(
    method=method, seed=seed, data=data_name, fairmetric=fairmetric)
if not os.path.isdir(path):
    os.makedirs(path)
logpath = open(os.path.join(path, "log.log"), "w")
sys.stdout = logpath
sys.stderr = logpath

fairness_dataset = FairnessDataset(data_source=data_source, random_state=seed, dataset_name=dataset_name)
X_train, y_train = fairness_dataset.X_train, fairness_dataset.y_train
X_test, y_test = fairness_dataset.X_test, fairness_dataset.y_test

fairness_info = {"fair_metric": fairmetric, "sensitive_attr": fairness_dataset.sensitive_attr}

if method == "Lexico":
    lexico_info = {}
    lexico_info["version"] = 1
    lexico_info["metric_priority"] = ["val_loss", "fairness"]
    lexico_info["tolerance"] = {"val_loss": 0.05, "fairness": 0.0}
    lexico_info["target"] = {"val_loss": 0.0, "fairness": 0.0}
elif method == "CFO":
    lexico_info = None
elif method == "constraint":
    lexico_info = None
else:
    lexico_info = {}
    lexico_info["version"] = 2
    lexico_info["metric_priority"] = ["val_loss", "fairness"]
    lexico_info["tolerance"] = {"val_loss": 0.05, "fairness": 0.0}
    lexico_info["target"] = {"val_loss": 0.0, "fairness": 0.0}

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
search_space = {
    'max_leaves': raytune.lograndint(4, upper),
    'max_depth': raytune.choice([0, 6, 12]),
    "n_estimators": raytune.lograndint(4, upper),
    "min_child_weight": raytune.loguniform(0.001, 128),
    "learning_rate": raytune.loguniform(1 / 1024, 1.0),
    "subsample": raytune.uniform(0.1, 1.0),
    "colsample_bytree": raytune.uniform(0.01, 1.0),
    "colsample_bylevel": raytune.uniform(0.01, 1.0),
    "reg_alpha": raytune.loguniform(1 / 1024, 1024),
    "reg_lambda": raytune.loguniform(1 / 1024, 1024),
    "learner": "xgboost",
}

# ---------------------------- val_loss ---------------------------------
algo = CFO(
    lexico_info=lexico_info,
    space=search_space,
    metric="val_loss",
    mode="min",
    seed=seed,
    low_cost_partial_config=low_cost_partial_config,
    points_to_evaluate=points_to_evaluate,
)
analysis = tune.run(
    evaluation_function,
    without_live_model=True,
    local_dir="logs/",
    num_samples=100000000,
    time_budget_s=budget / 2,
    search_alg=algo,
    use_ray=False,
    metric="val_loss",
)
best_result = analysis.best_result["val_loss"]
metrics_constraint = [("val_loss", "<=", best_result + 0.05)]
analysis = analysis.results
keys = list(analysis.keys())
length = len(keys)
objectives = ["val_loss", "fairness", "wall_clock_time", "config"]
histories = defaultdict(list)
for time_index in range(length):
    for objective in objectives:
        histories[objective].append(analysis[keys[time_index]][objective])

# ---------------------------- fairness ---------------------------------
algo = CFO(
    lexico_info=lexico_info,
    space=search_space,
    metric="fairness",
    mode="min",
    seed=seed,
    low_cost_partial_config=low_cost_partial_config,
    points_to_evaluate=points_to_evaluate,
)
analysis = tune.run(
    evaluation_function,
    without_live_model=True,
    local_dir="logs/",
    num_samples=100000000,
    time_budget_s=budget / 2,
    search_alg=algo,
    use_ray=False,
    metric_constraints=metrics_constraint,
    metric="fairness",
)
analysis = analysis.results
keys = list(analysis.keys())
length = len(keys)
objectives = ["val_loss", "fairness", "wall_clock_time", "config"]
for time_index in range(length):
    for objective in objectives:
        histories[objective].append(analysis[keys[time_index]][objective])
length = len(histories["val_loss"])
# ---------------------------- fairness ---------------------------------

for i in range(length):
    print("---------------------------")
    print("tiral", i)
    print("val_loss", histories["val_loss"][i])
    print("fairness", histories["fairness"][i])
    print("config", histories["config"][i])
    print("---------------------------")

savepath = os.path.join(path, "result.pckl")
print(savepath)
f = open(savepath, "wb")
pickle.dump(histories, f)
f.close()
logpath.close()

#  python fair_constraint.py --seed 1 --data german --fairmetric DSP --budget 5 --method CFO --method CFO
#  python fair_constraint.py --seed 1 --data german --fairmetric DSP --budget 5 --method CFO --method Lexico
