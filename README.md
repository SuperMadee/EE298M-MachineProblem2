# Machine Problem 2: Polynomial Solver using SGD and Tinygrad Framework

### Problem:
SGD is a useful algorithm with many applications. In this assignment, we will use SGD in the TinyGrad framework as polynomial solver - to find the degree and coefficients. The application will be used as follows: *python3 solver.py*

The solver will use data_train.csv to estimate the degree and coefficients of a polynomial. To test the generalization of the learned function, it should have small test error on data_test.csv. The function should be modeled using tinygrad : https://github.com/geohot/tinygrad
Use SGD to learned the polynomial coefficients.

***
### Background:

Here are some of the topics/terms that is relevant in this machine problem:

- **Tinygrad Framework**:
I solved this problem using **tinygrad** framework. Tinygrad is a deep learning framework that is small and has a sub 1000 locs. It is capable of various deep learning and machine learning applications such as GPU and Accelerator Support, YoLo, Stable Diffusion and etc. It was created by George Hotz, also known as geohotz, who became famous because of his self-driving cars and lawsuit of Sony from jailbreaking PS3.

- **Stochastic Gradient Descent (SGD)**
Stochastic gradient descent is an optimization algorithm often used in machine learning applications to find the model parameters that correspond to the best fit between predicted and actual outputs. Unlike gradient descent, SGD uses random samples from dataset (small mini-batches) to compute for the gradient for n updates

***
### What you need to know:

- **Installation**: I followed the installation guide from https://github.com/geohot/tinygrad

```
git clone https://github.com/geohot/tinygrad.git
cd tinygrad
python3 setup.py develop
```

- **Creating the virtual environment**: to ideally separate different applications and avoid problems with different dependencies.

```
conda create -n tgenv python=3.8
```

- **Install requirements.txt**
```
pip install -r requirements.txt
```
> Note: The following files should be inside the tinygrad folder in order to run smoothly:
> - requirements.txt
> - data_train.csv
> - data_test.csv
> - solver.py
>
>  The solver notebook was also included in this repository.

***
### Sample Output:
You may check the demo video here: https://bit.ly/Demo-MP2

***
### References:
- EE 298M Lecture Notes by Prof. Rowel Atienza, PhD.
- Tinygrad by Geohot (https://github.com/geohot/tinygrad)

***
### Acknowledgment:
I would like to thank Ma'am Izza and Sir Mark for their endless and support in guiding me to complete this Machine Problem.
