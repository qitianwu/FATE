## FeATure Extrapolation Networks (FATE)

Codes and datasets for [Towards Open-World Feature Extrapolation: An Inductive Graph Learning Approach](https://arxiv.org/abs/2110.04514).
This paper proposes a graph-based approach for handling new unseen features in test set. To verify the approach, we consider experiments on six small datasets from UCI repository and two large CTR prediction datasets.

### Dependency

Python 3.8, Pytorch 1.7, Pytorch Geometric 1.6

### Results

The experimental results on UCI datasets are shown below where our model FATE achieves superior testing performance.

![image](https://user-images.githubusercontent.com/22075007/158828734-f5f04f03-9cc7-4a90-9b16-195861916b8c.png)

The following table shows testing ROC-AUC results on two large-scale advertisement click-through rate prediction datasets Criteo and Avazu.

![image](https://user-images.githubusercontent.com/22075007/158828777-2930a148-b167-40b9-af97-985d57454535.png)


### Datasets

One can directly use our preprocessed UCI and CTR datasets for experiments, which we provide in the following link (download to the ***data*** folder)

    https://drive.google.com/drive/folders/1MlP5MiGeGNjb9GpWbI3HlUrpCFw2XqVA?usp=sharing
    
For each dataset of UCI, the instances are randomly splitted into 60/20/20% for train/valid/test and the 0-1 features are randomly divided into observed and unobserved ones. The ratios for observed features range from 30% to 80%. For example, the file "split_0.6_0.3.pkl" contains data splits with 60% training instances and 30% observed features. For more information above preprocessing, please refer to our paper.

For two CTR datasets, the instances are splitted according to time order. In specific, we split all the instances into 10 sets and use 1/1/8 sets for training/validation/testing. Such a splitting naturally introduce new unseen features in the validation and testing sets.

The original UCI datasets are collected from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets.php)

For the CTR prediction datasets, the original data are downloaded from Kaggle website. Speficially,

- Criteo http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/

- Avazu https://www.kaggle.com/c/avazu-ctr-prediction

### Model Implementation

The model contains a GNN module (for extrapolation) and a feedforward backbone. For UCI, we consider a shallow neural network as the backbone. For CTR, we consider both neural network and DeepFM model as the backbone.

### Run

To run the code, please refer to the bash script in each folder.

If you use the code or preprocessed datasets, please cite our paper:

```bibtex
    @inproceedings{wu2021fate,
    title = {Towards Open-World Feature Extrapolation: An Inductive Graph Learning Approach},
    author = {Qitian Wu and Chenxiao Yang and Junchi Yan},
    booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
    year = {2021}
    }
```
