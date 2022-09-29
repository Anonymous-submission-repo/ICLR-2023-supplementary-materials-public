from flaml.model import MyEstimator
from flaml import AutoML, CFO, tune
from collections import defaultdict
import pandas as pd
import argparse
import pickle
import sys
import os
import arff
import numpy as np
from ray import tune as raytune
import sys
from flaml.data import load_openml_dataset

data_func = {"german": 31, "ionosphere": 59, "vehicle": 54, "wine": 187, "zoo": 62, "christine": 41142,
             "adult": 179, "madelon": 1485, "scene": 312, "speech": 40910, "ginal": 1038, "nomao": 1486, "tecator": 851, "hill": 1479, "guil": 41159, "gina_prior": 1042, "gisette": 41026, "gina_agnostic": 1038, "bio": 4134, "hiva": 1039}

parser = argparse.ArgumentParser()
parser.add_argument("--seed", help="seeds", type=int, nargs="+")
parser.add_argument("--data", help="data", type=str, default="adult")
parser.add_argument("--budget", help="budget", type=int, default=3600)
parser.add_argument("--tolerance", help="tolerance", type=float, default=0.01)
parser.add_argument("--featurenum", help="feature number", type=int, default=100)
parser.add_argument("--method", help="method", type=str)
parser.add_argument("--setting", help="method", type=str)

args = parser.parse_args()
data_name = args.data
seed = args.seed[0]
budget = args.budget
dataset_name = args.data
method = args.method

if dataset_name == "gisette":
    alldata = arff.load(open('/workspaces/FLAML/gisette.arff', 'r'), encode_nominal=True)
    df = pd.DataFrame(alldata["data"])
    data_length = int(df.shape[0])
    X_train = df.iloc[0:int(data_length * 0.7), 0:-1]
    y_train = df.iloc[0:int(data_length * 0.7), -1]
    X_test = df.iloc[int(data_length * 0.7) + 1:-1, 0:-1]
    y_test = df.iloc[int(data_length * 0.7) + 1:-1, -1]
elif dataset_name == "zoo":
    alldata = arff.load(open('/workspaces/FLAML/zoo.arff', 'r'), encode_nominal=True)
    df = pd.DataFrame(alldata["data"])
    data_length = int(df.shape[0])
    X_train = df.iloc[0:int(data_length * 0.7), 0:-1]
    y_train = df.iloc[0:int(data_length * 0.7), -1]
    X_test = df.iloc[int(data_length * 0.7) + 1:-1, 0:-1]
    y_test = df.iloc[int(data_length * 0.7) + 1:-1, -1]
else:
    X_train, X_test, y_train, y_test = load_openml_dataset(dataset_id=data_func[dataset_name], data_dir="./download/")
if data_name in ["tecator", "speech", "musk", "kdd", "guil", "ionosphere", "gisette"]:
    from sklearn.preprocessing import MinMaxScaler
    scaler1 = MinMaxScaler()
    X_train = scaler1.fit_transform(X_train)
    scaler2 = MinMaxScaler()
    X_test = scaler2.fit_transform(X_test)
metric = "real_accuracy"
task = "classification"

if method in ["Lexico","new"]:
    path = "/workspaces/FLAML/result/overfitting-forest/{method}-seed_{seed}_data_{data}_tolerance_{tolerance}_setting_{setting}_budget_{budget}/".format(
        method=args.method, seed=seed, data=dataset_name, tolerance=args.tolerance, setting=args.setting, budget = budget)
elif method == "single":
    path = "/workspaces/FLAML/result/overfitting-forest/single-seed_{seed}_data_{data}_setting_{setting}_budget_{budget}/".format(
        method=args.method, seed=seed, data=dataset_name, setting=args.setting, budget = budget)
else:
    path = "/workspaces/FLAML/result/overfitting-forest/CFO-seed_{seed}_data_{data}_featurenum_{feature_num}_setting_{setting}_budget_{budget}/".format(
        seed=seed, data=dataset_name, feature_num=args.featurenum, setting=args.setting, budget = budget)
if not os.path.isdir(path):
    os.makedirs(path)
logpath = open(os.path.join(path, "log.log"), "w")
sys.stdout = logpath
sys.stderr = logpath

automl = AutoML()
automl.add_learner(learner_name="Myforest", learner_class=MyEstimator)
automl._state.eval_only = True
automl._state.fairness_info = None
settings = {
    "time_budget": 0,
    "metric": metric,
    "estimator_list": [
        "Myforest"
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
    lexico_info["metric_priority"] = ["val_loss", "feature_num"]
    lexico_info["tolerance"] = {"val_loss": args.tolerance, "feature_num": 0.0}
    lexico_info["target"] = {"val_loss": 0.0, "feature_num": 0.0}
elif method == "CFO":
    lexico_info = None
else:
    lexico_info = {}
    lexico_info["version"] = 2
    lexico_info["metric_priority"] = ["val_loss", "feature_num"]
    lexico_info["tolerance"] = {"val_loss": args.tolerance, "feature_num": 0.0}
    lexico_info["target"] = {"val_loss": 0.0, "feature_num": 0.0}
if data_name == "christine":
    feature_upper = 1611
elif data_name == "gina_prior":
    feature_upper = 632
elif data_name == "ionosphere":
    feature_upper = 33
elif data_name == "gisette":
    feature_upper = 4943
else:
    feature_upper = X_train.shape[1]
nrows = int(X_train.shape[0])
upper = min(2048, nrows)
init = 1 / np.sqrt(X_train.shape[1])
lower = min(0.1, init)
if method in ["CFO"]:
    selected_feature_num = 1611 if (data_name == "christine" and args.featurenum == 100) else int(
        np.median([1, int((args.featurenum / 100) * (X_train.shape[1])), X_train.shape[1]]))
else:
    selected_feature_num = None

search_space = {
    'n_estimators': raytune.lograndint(4, max(5, upper)),
    "max_leaves": raytune.lograndint(4, max(5, min(32768, nrows >> 1))),
    "feature_num": raytune.randint(1, feature_upper) if method in ["Lexico", "single","new"] else selected_feature_num,
    "max_features": raytune.loguniform(lower, 1.0),
    "learner": "Myforest",
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
                           "max_features": init,
                           }]
else:
    points_to_evaluate = [{"n_estimators": 4,
                           "max_leaves": 4,
                           "max_features": init,
                           "feature_num": feature_upper,
                           }]

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
analysis = analysis.results

keys = list(analysis.keys())
length = len(keys)
objectives = ["val_loss", "feature_num", "wall_clock_time", "config"]
histories = defaultdict(list)
for time_index in range(length):
    for objective in objectives:
        histories[objective].append(analysis[keys[time_index]][objective])

length = len(histories["val_loss"])
for i in range(length):
    print("---------------------------")
    print("tiral", i)
    print("feture_number", histories["feature_num"][i])
    print("val_loss", histories["val_loss"][i])
    print("config", histories["config"][i])
    print("---------------------------")

savepath = os.path.join(path, "result.pckl")
f = open(savepath, "wb")
pickle.dump(histories, f)
f.close()
logpath.close()

# python overfitting_forest.py --method Lexico --seed 1 --data german  --budget 3600 --tolerance 0.001
# python overfitting_forest.py --method single --seed 1 --data german  --budget 3600
# python overfitting_forest.py --method CFO --seed 1 --data german  --budget 3600  --featurenum 100
