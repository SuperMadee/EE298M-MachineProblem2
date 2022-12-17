#*~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~*
# Course: EE 298M - Foundations of Machine Learning
# Title: Machine Problem 2 - Polynomial Solver using SGD and Tinygrad 
# Framework

# Name: Ma. Madecheeen S. Pangaliman
# Student Number: 202220799
#*~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~**~*

# In order to read the data train and data test csv files, 
# the pandas library should be imported.
import pandas as pd

# Store the data train csv file in the df2 variable
df2 = pd.read_csv('data_train.csv')

# Import the Tensor module from tinygrad
from tinygrad.tensor import Tensor

# In order to read the contents of every tensor, import the numpy library
import numpy as np

# Pre-allocation of the x and y values of the data train csv file
data_train_x = np.array(df2.x)
data_train_y = np.array(df2.y)

# Transforming the x and y values into Tensors
data_train_x_tensor = Tensor([data_train_x])
data_train_y_tensor = Tensor([data_train_y])

# Creating the class Polysolver that contains the self and the forward
class PolySolver:
  def __init__(self,n):
    self.l1 = Tensor.uniform(1,n+1) # Initial guess for coefficients
    self.n = n # Degree

# Forward pass function. Will be used to calculate the output of the model
# using the input and the guess coefficients
  def forward(self, x):
    x_train = x**self.n
    
    for i in range(self.n-1,-1,-1):
      x_train = Tensor.cat(x_train,x**i)

    y_pred = self.l1@x_train
    return y_pred

# Setting the epochs, learning rate and the pre-allocation of coefficients, degree and losses
epochs = 5000
lr = 0.01
coeff = []
degree = []
loss_list = []

# Instantiation of the model and the optimizer per degree from 1 to 4 
# (Abel-Ruffini Theorem and Doc Atienza's hint)
for i in range(1,5):
  import tinygrad.nn.optim as optim
  model = PolySolver(i)
  optim = optim.SGD([model.l1], lr=lr)

# Storing the x and y training Tensors into X and Y variables for brevity purposes

  X = data_train_x_tensor
  Y = data_train_y_tensor

# Min-Max Scaler to scale down the value  
  X= ((X-X.min())/(X.max()-X.min()))
  Y= ((Y-Y.min())/(Y.max()-Y.min()))
  
  losslist = [] #to store losses

# Training the Model  
  for ep in range(epochs+1):
    preds = model.forward(X) #forward pass
    loss = ((preds-Y)**2).mean() #loss function - MSE
    optim.zero_grad() #zeroing gradients
    loss.backward() #backward pass
    optim.step() #parameter updates
    
# Print the degree, epochs and losses every after 500 iterations
    if ep%500 == 0:
      print("Degree: {}, Epoch: {}, loss: {}".format(i, ep, loss.data))
      losslist.append(loss.data)
  
  coeff.append(model.l1.numpy())
  degree.append(model.l1.shape[1]-1)
  loss_list.append(loss.numpy())


# Locates the values of the coefficients and degree with the least loss and prints it
locator = loss_list.index(min(loss_list))
print("The degree is {} and the coefficients are {} with a loss of {}".format(degree[locator], coeff[locator], float(loss_list[locator])))


# Testing the model
# Store the data test csv file in the df3 variable
df3 = pd.read_csv('data_test.csv')

# Pre-allocation of the x and y values of the data train csv file
data_test_x = np.array(df3.x)
data_test_y = np.array(df3.y)

# Transforming the values into Tensors
data_test_x_tensor = Tensor([data_test_x])
data_test_y_tensor = Tensor([data_test_y])

# Degree of the locator
m = int(degree[locator])

# Storing the x and y test Tensors into Xt and Yt variables for brevity purposes
Xt = data_test_x_tensor
Yt = data_test_y_tensor

# Min-Max Scaler
Xt= ((Xt-Xt.min())/(Xt.max()-Xt.min()))
Yt= ((Yt-Yt.min())/(Yt.max()-Yt.min()))

# Calculating the error between the actual and predicted value for y 
x_test = Xt**m
coefft = Tensor(coeff[locator])
  
for i in range(m-1,-1,-1):
   x_test = Tensor.cat(x_test,Xt**i)

y_test = coefft@x_test
loss_test = ((y_test-Yt)**2).mean()

# Prints the loss incurred when using the test data
print("The lost for the test data is {}" .format(float(loss_test.numpy())))