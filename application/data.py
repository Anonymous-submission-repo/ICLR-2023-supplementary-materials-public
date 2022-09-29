import numpy as np
import pandas as pd
# logger = logging.getLogger(__name__)

class FairnessDataset:
    """
    A class for the dataset
    """

    def __init__(self,
                 data_source='openml',
                 dataset_id=1590,
                 dataset_name='adult',
                 random_state=0,
                 dataset_format='dataframe',
                 sensitive_attr=None,
                 revised_targets={}
                 ):
        self.X_train = self.X_test = self.y_train = self.y_test = None
        # get the original dataset and splict them into train and test
        if data_source == 'openml':
            # Get datasets from openml
            if dataset_name == 'adult':
                sensitive_attr = 'sex' if sensitive_attr is None else sensitive_attr
                revised_targets = {'>50K': 1, '<=50K': 0}
                dataset_id = 1590
                from flaml.data import load_openml_dataset
                self.X_train, self.X_test, self.y_train, self.y_test = load_openml_dataset(
                    dataset_id=dataset_id, data_dir='./', random_state=random_state, dataset_format=dataset_format)
            else:
                raise NotImplementedError
        elif data_source == 'aif360':
            dataset_orig = None
            # Get datasets from AIF360
            from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import \
                load_preproc_data_adult, load_preproc_data_compas, load_preproc_data_german
            if 'adult' in dataset_name:
                dataset_orig = load_preproc_data_adult()
            elif 'compas' in dataset_name:
                dataset_orig = load_preproc_data_compas()
            elif 'german' in dataset_name or 'credit' in dataset_name:
                dataset_orig = load_preproc_data_german()
                print('metadata', dataset_orig.metadata)
                print('data set type', type(dataset_orig))
            elif 'bank' in dataset_name:
                from aif360.datasets.bank_dataset import BankDataset
                dataset_orig = BankDataset()
            elif 'law' in dataset_name:
                from aif360.sklearn.datasets import fetch_lawschool_gpa
                from sklearn import preprocessing
                # load and perform pre-processing following instructions from
                # https://github.com/Trusted-AI/AIF360/blob/746e763191ef46ba3ab5c601b96ce3f6dcb772fd/examples/sklearn/demo_grid_search_reduction_regression_sklearn.ipynb
                
                # 1. load the law school gpa dataset from tempeh
                X_train, y_train = fetch_lawschool_gpa(subset="train")
                X_test, y_test = fetch_lawschool_gpa(subset="test")
                X_train.head()
                # 2. map the protected attributes to integers
                X_train.index = pd.MultiIndex.from_arrays(X_train.index.codes, names=X_train.index.names)
                X_test.index = pd.MultiIndex.from_arrays(X_test.index.codes, names=X_test.index.names)
                y_train.index = pd.MultiIndex.from_arrays(y_train.index.codes, names=y_train.index.names)
                y_test.index = pd.MultiIndex.from_arrays(y_test.index.codes, names=y_test.index.names)
                # 3. use Pandas for one-hot encoding for easy reference to columns associated with proected 
                # attribute
                X_train, X_test = pd.get_dummies(X_train), pd.get_dummies(X_test)
                # 4. normalize the continuous values
                
                min_max_scaler = preprocessing.MinMaxScaler()
                X_train = pd.DataFrame(min_max_scaler.fit_transform(X_train.values),columns=list(X_train),index=X_train.index)
                X_test = pd.DataFrame(min_max_scaler.transform(X_test.values),columns=list(X_test),index=X_test.index)
                # prot_attr_cols = [col for col in list(X_train) if "race" in col]
                # prot_attr_cols_test = [col for col in list(X_test) if "race" in col]
                # X_train_dropped = X_train.drop(prot_attr_cols,axis=1)
                # X_test_dropped = X_test.drop(prot_attr_cols_test,axis=1)
                # TODO: should we drop the sensitive attributes?
                # We are not dropping it in classification tasks but here in this notebook,
                # they are  dropping the sensitive attribute
                self.X_train = X_train
                self.X_test = X_test
                # self.X_train = X_train_dropped
                # self.X_test = X_test_dropped
                self.y_train = y_train
                self.y_test = y_test
                sensitive_attr = 'race_black'
            else:
                print('dataset not available, please provide a dataset among [adult, german, compas]')
                raise ValueError
            # privileged_groups = [{sensitive_attr: 1}]
            # unprivileged_groups = [{sensitive_attr: 0}]
            if self.X_train is None:
                np.random.seed(0)
                df_dataset, _ = dataset_orig.convert_to_dataframe()
                dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)
                df_train, _ = dataset_orig_train.convert_to_dataframe()
                df_test, _ = dataset_orig_test.convert_to_dataframe()
                self.X_train, self.y_train = df_train[dataset_orig_train.feature_names], df_train[dataset_orig_train.label_names]
                self.X_test, self.y_test = df_test[dataset_orig_train.feature_names], df_test[dataset_orig_train.label_names]
            for df in (self.X_train, self.y_train, self.X_test, self.y_test):
                try:
                    df.columns = df.columns.str.replace("<", "_")
                except AttributeError:
                    print('object has no attribute columns')
            self.y_train = self.y_train.squeeze()
            self.y_test = self.y_test.squeeze()
            # converting label from {1.0, 2.0} to {1.0, 0.0}
            # Reason: the default label mapping in dataset german is
            # {'label_maps': [{1.0: 'Good Credit', 0.0: 'Bad Credit'}],}
            # fairlearn only support binary label {-1,1} or {0, 1}.
            self.y_test = self.y_test.replace(to_replace=2.0, value=0.0)
            self.y_train = self.y_train.replace(to_replace=2.0, value=0.0)
            # use the first protected attribute by default
            sensitive_attr = dataset_orig.protected_attribute_names[0] if (dataset_orig is not None and sensitive_attr is None) else sensitive_attr
        elif data_source == 'fairbo_credit':
            from sklearn.model_selection import train_test_split
            credit_df = process_german()
            data_length = int(credit_df.shape[0])
            X_all = credit_df.iloc[0:data_length, 0:-1]
            y_all = credit_df.iloc[0:data_length, -1]
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X_all, y_all, test_size=0.3, random_state=random_state)
        else:
            raise NotImplementedError
        if revised_targets:
            self.y_test = self.y_test.replace(revised_targets)
            self.y_train = self.y_train.replace(revised_targets)
        self.sensitive_attr = sensitive_attr
        self.revised_targets = revised_targets

def process_german():
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data', header=None, sep=' ')
    df.columns = ["CheckingAC_Status", "MaturityMonths", "CreditHistory", "Purpose", "LoanAmount", "SavingsAC",
                  "Employment", "InstalmentPctOfIncome", "SexAndStatus", "OtherDebts", "PresentResidenceYears",
                  "Property", "Age", "OtherInstalmentPlans", "Housing", "NumExistingLoans", "Job",
                  "Dependents", "Telephone", "ForeignWorker", "Class1Good2Bad"]

    df["target"] = df["Class1Good2Bad"].replace([1, 2], [1, 0]).astype("category")
    df = df.drop(columns=["Class1Good2Bad"])
    df["CheckingAC_Status"] = (
        df["CheckingAC_Status"]
        .replace(["A11", "A12", "A13", "A14"], ["x < 0 DM", "0 <= x < 200 DM", "x >= 200DM", "no checking account"])
        .astype("category")
    )
    df["CreditHistory"] = (
        df["CreditHistory"]
        .replace(
            ["A30", "A31", "A32", "A33", "A34"],
            ["no credits", "all credits paid", "existing credits paid", "delay", "critical accnt. / other credits"],
        )
        .astype("category")
    )
    df["Purpose"] = (
        df["Purpose"]
        .replace(
            ["A40", "A41", "A42", "A43", "A44", "A45", "A46", "A47", "A48", "A49", "A410"],
            [
                "new car",
                "used car",
                "forniture",
                "radio/tv",
                "appliances",
                "repairs",
                "education",
                "vacation",
                "retraining",
                "business",
                "others",
            ],
        )
        .astype("category")
    )
    df["SavingsAC"] = (
        df["SavingsAC"]
        .replace(
            ["A61", "A62", "A63", "A64", "A65"],
            ["x < 100 DM", "100 <= x < 500 DM", "500 <= x < 1000 DM", "x >= 1000 DM", "unknown"],
        )
        .astype("category")
    )
    df["Employment"] = (
        df["Employment"]
        .replace(
            ["A71", "A72", "A73", "A74", "A75"],
            ["unemployed", "x < 1 year", "1 <= x < 4 years", "4 <= x < 7 years", "x >= 7 years"],
        )
        .astype("category")
    )
    df["SexAndStatus"] = (
        df["SexAndStatus"]
        .replace(
            ["A91", "A92", "A93", "A94", "A95"],
            [
                "male divorced/separated",
                "female divorced/separated/married",
                "male single",
                "male married/widowed",
                "female single",
            ],
        )
        .astype("category")
    )
    df["OtherDebts"] = (
        df["OtherDebts"].replace(["A101", "A102", "A103"], ["none", "co-applicant", "guarantor"]).astype("category")
    )
    df["Property"] = (
        df["Property"]
        .replace(
            ["A121", "A122", "A123", "A124"],
            ["real estate", "soc. savings / life insurance", "car or other", "unknown"],
        )
        .astype("category")
    )
    df["OtherInstalmentPlans"] = (
        df["OtherInstalmentPlans"].replace(["A141", "A142", "A143"], ["bank", "stores", "none"]).astype("category")
    )
    df["Housing"] = df["Housing"].replace(["A151", "A152", "A153"], ["rent", "own", "for free"]).astype("category")
    df["Job"] = (
        df["Job"]
        .replace(
            ["A171", "A172", "A173", "A174"],
            [
                "unemployed / unskilled-non-resident",
                "unskilled-resident",
                "skilled employee / official",
                "management / self-employed / highly qualified employee / officer",
            ],
        )
        .astype("category")
    )
    df["Telephone"] = df["Telephone"].replace(["A191", "A192"], ["none", "yes"]).astype("category")
    df["ForeignWorker"] = df["ForeignWorker"].replace(["A201", "A202"], ["yes", "no"]).astype("category")
    return df