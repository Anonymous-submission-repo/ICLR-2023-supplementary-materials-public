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

data_func = {"brest": 1165, "colon": 1137}
parser = argparse.ArgumentParser()
parser.add_argument("--seed", help="seeds", type=int, nargs="+")
parser.add_argument("--data", help="data", type=str, default="adult")
parser.add_argument("--budget", help="budget", type=int, default=3600)
parser.add_argument("--method", help="method", type=str)
parser.add_argument("--setting", help="method", type=str, default="both")
args = parser.parse_args()
data_name = args.data
seed = args.seed[0]
budget = args.budget/2
dataset_name = args.data
method = args.method
X_train, X_test, y_train, y_test = load_openml_dataset(dataset_id=data_func[dataset_name], data_dir="./download/",random_state = seed)
metric = "real_accuracy"
task = "classification"
path = "/workspaces/FLAML/result/bio001/{method}-seed_{seed}_data_{data}_setting_{setting}/".format(
    method=method, seed=seed, data=dataset_name, setting=args.setting)

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

if method == "Lexico":
    lexico_info = {}
    lexico_info["version"] = 1
    lexico_info["metric_priority"] = ["val_loss", "feature_num", "stability"]
    lexico_info["tolerance"] = {"val_loss": 0.01, "feature_num": 0.0, "stability": 0.0}
    lexico_info["target"] = {"val_loss": 0.0, "feature_num": 0.0, "stability": 500}
elif method == "CFO":
    lexico_info = None
elif method == "constraint":
    lexico_info = None
else:
    lexico_info = {}
    lexico_info["version"] = 2
    lexico_info["metric_priority"] = ["val_loss", "feature_num", "stability"]
    lexico_info["tolerance"] = {"val_loss": 0.01, "feature_num": 0.0, "stability": 0.0}
    lexico_info["target"] = {"val_loss": 0.0, "feature_num": 0.0, "stability": 500}

feature_upper = X_train.shape[1]
upper = max(5, min(32768, int(X_train.shape[0])))
# if method in ["CFO"]:
#     selected_feature_num = 1611 if (data_name == "christine" and args.featurenum == 100) else int(
#         np.median([1, int((args.featurenum / 100) * (X_train.shape[1])), X_train.shape[1]]))
# else:
selected_feature_num = None
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
    "feature_num": raytune.randint(1, feature_upper),
    "learner": "my_xgboost",
    "get_stable": True,
}

low_cost_partial_config = {
    "n_estimators": 4,
    "max_leaves": 4,
    "feature_num": 4
} if args.setting in ["lowstart", "both"] else {
    "n_estimators": 4,
    "max_leaves": 4,
}
if args.setting in ["lowstart", "normal"]:
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
else:
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
                           "feature_num": feature_upper,
                           }]

# ----------------- 1 obj -----------------------------
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
    local_dir=path,
    num_samples=100000000,
    time_budget_s=budget,
    search_alg=algo,
    use_ray=False,
    metric="val_loss",
    log_file_name="log_flaml",
)
best_result = analysis.best_result["val_loss"]
analysis = analysis.results
keys = list(analysis.keys())
length = len(keys)
# print("length1",length)
objectives = ["val_loss", "feature_num", "wall_clock_time", "config", "stability"]
histories = defaultdict(list)
for time_index in range(length):
    for objective in objectives:
        histories[objective].append(analysis[keys[time_index]][objective])
print(histories["val_loss"])
print(histories["wall_clock_time"])
# ----------------- 2 obj -----------------------------
algo = CFO(
    lexico_info=lexico_info,
    space=search_space,
    metric="stability",
    mode="min",
    seed=seed,
    low_cost_partial_config=low_cost_partial_config,
    points_to_evaluate=points_to_evaluate,
)
metrics_constraint = [("val_loss", "<=", best_result + 0.01), ("feature_num", "<=", 500)]
analysis = tune.run(
    evaluation_function,
    without_live_model=True,
    local_dir=path,
    num_samples=100000000,
    time_budget_s=budget,
    search_alg=algo,
    use_ray=False,
    metric="stability",
    metric_constraints=metrics_constraint,
    log_file_name="log_flaml",
)
analysis = analysis.results
keys = list(analysis.keys())
length = len(keys)
objectives = ["val_loss", "feature_num", "wall_clock_time", "config", "stability"]
for time_index in range(length):
    for objective in objectives:
        histories[objective].append(analysis[keys[time_index]][objective])
print(histories["val_loss"])
print(histories["wall_clock_time"])
# -----------------------------------------------------
length = len(histories["val_loss"])
# print("final_length", length)
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

# python bio.py --method Lexico --seed 1 --data german  --budget 3600
# python bio.py --method single --seed 1 --data german  --budget 3600
# python bio.py --method CFO --seed 1 --data german  --budget 3600
