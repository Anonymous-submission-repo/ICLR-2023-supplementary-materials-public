# Targeted Hyperparameter Optimization with Lexicographic Preferences Over Multiple Objectivesr

## **:fire: Notice**


### :bangbang:  We have integrated LexiFlow into AutomL Library [FLAML](https://github.com/microsoft/FLAML)! 

Please refer to the doc https://microsoft.github.io/FLAML/docs/Use-Cases/Tune-User-Defined-Function#lexicographic-objectives as well as an example in https://github.com/microsoft/FLAML/blob/main/notebook/tune_lexicographic.ipynb.

## Introduce
This repository is the implementation of **Targeted Hyperparameter Optimization with Lexicographic Preferences Over Multiple Objectives**. 

The implementation of our method LexiFlow is built upon an open-source AutoML library named FLAML. Thus the submitted code includes part of flaml’s code. But we emphasize that the contributors and copyright information about the open-source library FLAML do not necessarily reveal the identities of the authors of this work. We plan to open source the code accompanying the formal publication of this paper.

This version of the code is made to facilitate the peer review of the ICLR 2023 submission of our paper. 
We plan to release the code accompanying the formal publication of this paper. 


## Datasets
In tuning XGboost, we verify the performance of LexiFlow on the datasets shown below. All of these datasets are available on OpenML.

1. In tuning random forest and Xgboost, the datasets we use in our paper are all available in openml.
2. In tuning neural networks, we verify the performance of LexiFLOW on [FashionMnist](https://www.kaggle.com/datasets/zalando-research/fashionmnist) dataset.

## Experiments

### **Requirements**

To install requirements:
```setup
pip install -r requirements.txt
```


### **How to run** 

1. Fairness 
- LexiFLow
    ```
    python fair.py --seed 1 --data compas --fairmetric DSP --budget 3600 --method LexiFlow  
    ```
- Baseline
    ```
    python fair_mohpo.py --seed 1 --data compas --fairmetric DSP --budget 3600  
    ```
    ```
    python fair.py --seed 1 --data compas --fairmetric DSP --budget 3600 --method CFO 
    ```
    ```
    python fair_constraint.py --seed 1 --data compas --fairmetric DSP --budget 3600 --method constraint 
    ```

2. NN

- LexiFLow

    ```
    python nn_cfo.py --method LexiFlow --seed 1 --budget 7200 --second_obj params  
    ```
- Baselines

    ```
    python nn_cfo.py --method CFO --seed 1 --budget 7200 --second_obj params  
    ```
    ```
    python nn_constraint.py --method constraint --seed 1 --budget 7200 --second_obj params  
    ```
    ```
    python nn_mohpo.py --seed 1 --budget 3600 --second_obj params  
    ```


3. Bio

- LexiFLow

    ```
      python bio.py --method LexiFlow --seed 1 --data colon  --budget 10000
    ```
- Baselines

    ```
    python bio.py --method CFO --seed 1 --data colon  --budget 10000
    ```
    ```
    python bio.py --method single --seed 1 --data colon  --budget 10000
    ```
    ```
    python bio_mohpo.py --seed 1 --data colon --budget 3600 
    ```


4. overfitting

- LexiFLow

    ```
    python overfitting_forest.py --method LexiFlow --seed 1 --data christine  --budget 10000 --tolerance 0.001
    ```
- Baselines

    ```
    python overfitting_forest.py --method single --seed 1 --data christine  --budget 10000 --tolerance 0.001
    ```
    ```
    python overfitting_forest.py --method CFO --seed 1 --data christine  --budget 10000 --tolerance 0.001
    ```










