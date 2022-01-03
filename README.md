## This repository holds the implementation for paper [Towards Open-World Feature Extrapolation: An Inductive Graph Learning Approach](https://arxiv.org/abs/2110.04514)

Download our preprocessed UCI dataset to the ***data*** folder via the following link

    https://drive.google.com/drive/folders/1MlP5MiGeGNjb9GpWbI3HlUrpCFw2XqVA?usp=sharing

For each dataset, the instances are randomly splitted into 60/20/20% for train/valid/test and the 0-1 features are randomly divided into observed and unobserved ones. The ratios for observed features range from 30% to 80%. For example, the file "split_0.6_0.3.pkl" contains data splits with 60% training instances and 30% observed features. For more information, please refer to our paper.

For the CTR prediction datasets, please download them on Kaggle website.

To run the code, please refer to the bash script in each folder.

More information will be updated.

If you use the code or preprocessed datasets, please cite our paper:

    @inproceedings{wu2021fate,
    title = {Towards Open-World Feature Extrapolation: An Inductive Graph Learning Approach},
    author = {Qitian Wu and Chenxiao Yang and Junchi Yan},
    booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
    year = {2021}
    }
