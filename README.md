# Machine Problem 2: Polynomial Solver using SGD and Tinygrad Framework

### Problem:
SGD is a useful algorithm with many applications. In this assignment, we will use SGD in the TinyGrad framework as polynomial solver - to find the degree and coefficients. The application will be used as follows: *python3 solver.py*

The solver will use data_train.csv to estimate the degree and coefficients of a polynomial. To test the generalization of the learned function, it should have small test error on data_test.csv. The function should be modeled using tinygrad : https://github.com/geohot/tinygrad
Use SGD to learned the polynomial coefficients.

***
### What you need to know?

- *Installation*: I followed the installation guide from https://github.com/geohot/tinygrad

```
git clone https://github.com/geohot/tinygrad.git
cd tinygrad
python3 setup.py develop
```

- *Creating the virtual environment*: to ideally separate different applications and avoid problems with different dependencies.

```
conda create -n tgenv python=3.8
```

- *Install requirements.txt*
```
pip install -r requirements.txt
```
> Note: The following files should be inside the tinygrad folder in order to run smoothly:
> - requirements.txt
> - data_train.csv
> - data_test.csv
> - solver.py
> The solver notebook was also included in this repository.

