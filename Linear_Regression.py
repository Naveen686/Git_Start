#Importing Libraries

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


#Loadind Data and dividing into train and test sets (Salary_Data contains 30 exmaples)

data = pd.read_csv("Salary_Data.csv")
x_train = data.iloc[:25, 0].values
y_train = data.iloc[:25, 1].values
x_test = data.iloc[25:, 0].values
y_test = data.iloc[25:, 1].values


#Defining a function for Normalizing feature values

def feature_normalize(x):
    mean = np.mean(x)
    range_value = np.amax(x) - np.amin(x)
    x = x - mean
    x /= range_value
    return x
    
    
#Normalizing features in Train and Test data

x_train_norm = feature_normalize(x_train)
x_test_norm = feature_normalize(x_test)


#Function for Updating weight and bias using Gradient Descent

def update_w_b(x, y, w, b, alpha):
    
    dw = 0
    db = 0
    loss = 0
    n = len(x)
    for i in range(n):
        dw += -2*x[i]*(y[i] - (w*x[i] + b))
        db += -2*(y[i] - (w*x[i] + b))
        loss += (y[i] - (w*x[i] + b))**2
    loss = loss/n
    w = w - alpha * (dw/n)
    b = b - alpha * (db/n)
    return w, b, loss
    
    
#Function for training the data

def train(x, y, w, b, alpha, epoch):
    j = []
    for e in range(epoch+1):
        w, b, loss = update_w_b(x, y, w, b, alpha)
        j.append(loss)
        #print(w, b)
        if(e % 1000 == 0):
            print("loss at {} epoch is {}".format(e, loss))
            print("Learning parameteres are {}, {}".format(w, b))
    return w, b, j
    

#Training

epoch = 10000
w, b, j = train(x_train, y_train, 0, 0, 0.01, epoch)   


#Plotting loss w.r.t epoch

x = np.arange(epoch + 1)
plt.plot(x, j)
plt.title("Loss vs Epoch")
plt.xlabel("Epochs")
plt.ylabel("Loss")


#Plotting the line(with learnt parameters w & b) on training data

y = [None] * len(x_train)
for i in range(len(x_train)):
    y[i] = w*x_train[i]+b
plt.scatter(x_train, y_train)
plt.plot(x_train, y) 


#Predicting on Test data and finding error

pred_x_test = [None] * len(x_test)
for i in range(len(x_test)):
    pred_x_test[i] = w * x_test[i] + b 
plt.scatter(x_test, y_test)
plt.plot(x_test, pred_x_test)
error = 0
for i in range(len(x_test)):
    error +=abs(y_test[i]-pred_x_test[i])
error /= len(x_test)

