# CuckooSearch-Kmeans Algorithm

CuckooSearch-Kmeans algorithm (CS-Kmeans) is a learning framework to make recommendations. The key idea is to learn the user-item interaction using matrix factorization (MF) or neural collaborative filtering (NCF) to project them into a latent space and explore the latter to find the best clusterization to use as input a model-based collaborative filtering algorithm.

## Dataset

[ratings.csv](https://grouplens.org/datasets/movielens/100k/) is used to train and test the model and [movies.csv](https://grouplens.org/datasets/movielens/100k/) to get titles of recommended movies.

## Files

> `methods.py`: contains the cuckoo search, k-means, cuckoo-kmeans and relative cross validation
>
> `utils.py`: some handy functions for model training etc.
>
> `nnls.py` and `NonNegativeMatrixFactorization.py`: Block Principal Pivoting Method and Non Negative Matrix Factorization
>
> `function.py`: objective function environment
>
> `run_examples.ipynb`: runs cuckooo search on some test functions as Ackley and Himmelbau
>
> `run_mf.ipynb`: runs matrix factorization (MF), cross validating it and producing  `user_latent_matrix_mf.py`
>
> `run_ncf.ipynb`: runs neural collaborative filtering (NCF), cross validating it and producing `user_latent_matrix_ncf.py`
>
> `user_latent_matrix_mf.py` and `user_latent_matrix_ncf.py`: user latent matrices produced respectively by MF and NCF methods
>
> `main.ipynb`: runs the CS-Kmeans algorithm implemented with both MF and NCF and tests their accuracies
