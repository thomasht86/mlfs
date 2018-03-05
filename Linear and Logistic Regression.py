
# coding: utf-8

# # Linear and Logistic Regression

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# ## Implement linear regression with ordinary least squares (OLS) using the closed-form solution seen in 

# In[2]:


def linear_regression(x, y):
    '''
    Closed form linear regression . 
    '''
    weights = np.linalg.pinv(x.transpose().dot(x)).dot(x.transpose()).dot(y)
    return weights


# ## Load the data in train_2d_reg_data.csv (training data) and test_2d_reg_data.csv (test data) and use your OLS implementation to find a set of good weights for the training data. Show the weights as well as the model error $E_(mse)(w)$ for the training and test set after training. Is your model generalizing well?

# In[200]:


train = pd.read_csv("datasets/regression/train_2d_reg_data.csv", header=None).values
test = pd.read_csv("datasets/regression/test_2d_reg_data.csv", header=None).values


# In[201]:


def prep(data):
    '''
    Function to prep the data, by separating features and target, and adding an 
    intercept column to the features. 
    '''
    x = data[:,:-1]
    y = data[:, -1]
    # add intercept
    intercept = np.ones_like(x[:,0:1])
    x = np.concatenate([intercept, x], 1)
    return x, y


# In[202]:


x_train, y_train = prep(train)
x_test, y_test = prep(test)
x_train.shape, y_train.shape, x_test.shape, y_test.shape


# In[203]:


weights = linear_regression(x_train, y_train)


# In[204]:


weights


# In[205]:


def get_predictions(x, weights):
    '''
    Calculates the predictions for a dataset given 
    the calculated weights.
    '''
    return x.dot(weights)

def mse(y_preds, y_true):
    '''
    Caculates the mean squared error between
    the predictions and the ground truth.
    '''
    return np.mean((y_true-y_preds)**2)


# ### Get predictions for training set

# In[206]:


train_preds = get_predictions(x_train, weights)


# ### Calculate mse for training set

# In[207]:


mse(train_preds, y_train)


# ### Get predictions for test set

# In[208]:


test_preds = get_predictions(x_test, weights)


# ### Calculate mse for test set

# In[209]:


mse(test_preds, y_test)


# We can see that the model generalizes well to the test set. 

# ## Load the data in train_1d_reg_data.csv and test_1d_reg_data.csv and use your OLS implementation to find a set of good weights for the training data. Using these weights, make a plot of the line fitting the data and show this in the report. Does the line you found fit the data well? If not, discuss in broad terms how this can be remedied. 

# In[261]:


train = pd.read_csv("datasets/regression/train_1d_reg_data.csv", header=None).values
test = pd.read_csv("datasets/regression/test_1d_reg_data.csv", header=None).values


# In[262]:


x_train, y_train = prep(train)
x_test, y_test = prep(test)
x_train.shape, y_train.shape, x_test.shape, y_test.shape


# In[263]:


weights = linear_regression(x_train, y_train)


# In[264]:


weights


# In[265]:


preds = get_predictions(x_train, weights)


# In[269]:


plt.title("Data vs OLS-predictions")
plt.xlabel("x")
plt.ylabel("y")
plt.scatter(x_train[:,1], y_train)
plt.scatter(x_train[:,1], preds)


# We can see that the line fits our data well. 

# ## Logistic regression

# In[224]:


def sigmoid(z):
    '''
    Sigmoid function.
    '''
    return 1/(1+np.exp(-z))


# In[229]:


def update_weights(x, y, w, eta):
    '''
    Weight update for gradient descent for logistic regression.
    '''
    preds = sigmoid(x.dot(w))
    w = w - (1/len(x))*eta*np.dot(x.T, preds-y)
    return w


# In[226]:


def cross_entropy(x, y, w):
    '''
    The negative log-likelihood, or cross-entropy error function.
    '''
    preds = sigmoid(x.dot(w))
    ce = -1*(np.log(preds).T.dot(y)+np.log(1-preds).T.dot(1-y))
    return ce.sum()/len(y)


# In[227]:


def binarize(probs):
    '''
    Function to binarize a probability prediction. 
    Returns 1 ifp>0.5, else 0.
    '''
    return (probs>0.5).astype(int)


# In[228]:


def accuracy(preds, true):
    '''
    Vectorized function to evaluate accuracy, or fraction of correct credictions.
    '''
    return np.mean(preds == true)


# ### Load the data in cl_train_1.csv and cl_test_1.csv and use your logistic regression implementation to train on the data in the training set. Is the data linearly separable? Explain your reasoning. Additionally, show a plot of the cross-entropy error for both the training and test set over 1000 iterations oftraining. What learning rate η and initial parameters w did you select? Is your model generalising well?

# In[271]:


train = pd.read_csv("datasets/classification/cl_train_1.csv", header=None).values
test = pd.read_csv("datasets/classification/cl_test_1.csv", header=None).values


# In[272]:


x_train, y_train = prep(train)
x_test, y_test = prep(test)
x_train.shape, y_train.shape, x_test.shape, y_test.shape


# In[288]:


def logreg_train(x_train, y_train, x_test, y_test, lr=0.2, iterations=1000):
    '''
    Whole function to fit a logistic regression model to x_train and y_train. 
    We also save both training errors, testing errors, and accuracy, so we can 
    visualize the progression of the training. 
    '''
    weights = (2*np.random.random(x_train.shape[1]))-1
    tr_errors = []
    test_errors = []
    tr_a, test_a = [], []
    for i in range(iterations):
        weights = update_weights(x_train, y_train, weights, lr)
        tr_error = cross_entropy(x_train, y_train, weights)
        test_error = cross_entropy(x_test, y_test, weights)
        tr_errors.append(tr_error)
        test_errors.append(test_error)
        tr_acc = accuracy(binarize(sigmoid(x_train.dot(weights))), y_train)
        test_acc = accuracy(binarize(sigmoid(x_test.dot(weights))), y_test)
        tr_a.append(tr_acc)
        test_a.append(test_acc)
    return tr_errors, test_errors, tr_a, test_a, weights


# In[289]:


tr_errors, test_errors, tr_a, test_a, weights = logreg_train(x_train, y_train, x_test, y_test)


# In[277]:


def plot_errors(tr_errors, test_errors):
    plt.title("Training/testing error vs iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Cross-entropy error")
    plt.plot(tr_errors, label="Training error")
    plt.plot(test_errors, label="Test error")
    plt.legend()


# In[290]:


def plot_acc(tr_a, test_a):
    plt.title("Training/testing accuracy vs iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.plot(tr_a, label="Training accuracy")
    plt.plot(test_a, label="Test accuracy")
    plt.legend()


# In[278]:


plot_errors(tr_errors, test_errors)


# We can see that our model generalizes pretty well, but the test error is a bit higher than the training error. 

# In[291]:


plot_acc(tr_a, test_a)


# I chose a learning rate of 0.2, and randomly initialized the weights from an uniform distribution over $[-1,1)$

# To plot our decision boundary, we set 
# $w^Tx=0$ <br>
# That gives us the non-vectorized equation: <br>
# $w_0+w_1*x_0+w_2*x_1=0$ <br>
# By solving for x1, we get x_1 as a function of x_0, where $w^Tx=0$: <br>
# $x_1 = (-w_0-w_1*x_0)/w_2$ <br>
# This enables us to plot our decision boundary. 
# Note that we coud also represent our decision function as $\sigma(w^Tx)=0.5$ <br>
# Because $\sigma(0)=0.5$, this would be equivalent. 

# In[281]:


line = lambda x: (-weights[0]-weights[1]*x)/weights[2]


# In[282]:


weights


# In[285]:


def plot_lin_db(train, linefunc):
    plt.title("Logistic regression data and decision boundary")
    pos = plt.scatter(train[np.where(train[:,2]>0),0], train[np.where(train[:,2]>0), 1], )
    neg = plt.scatter(train[np.where(train[:,2]==0),0], train[np.where(train[:,2]==0), 1])
    db = plt.scatter([0.01*i for i in range(100)], [[linefunc(0.01*i) for i in range(100)]], s=10)
    plt.legend((pos, neg, db),('Positive', 'Negative', 'Decision Boundary'))


# I want to plot the decision boundary to visualize the results better. 
# First, we can plot the decision boundary together with the training data.

# In[286]:


plot_lin_db(train, line)


# We see that our model fits the training data perfectly, as our training accuracy also told us.  
# The training data is _linearly separable_, as we can see a line is able to fully separate the two classes.  
# 
# How well does the model generalize to the test data?  
# Lets make a plot of the models decision boundary along with the test data. 

# In[287]:


plot_lin_db(test, line)


# We can see that it is a pretty good fit, but not quite perfect, as one point is misclassified. 

# ### Load the data in cl_train_s2.csv and cl_test_2.csv and use your logistic regression implementation to train on the data in the training set. Is the data linearly separable? Explain your reasoning. Plot the decision boundary as explained in the previous task as well as the data points in the training and test set. Discuss what can be done for logistic regression to correctly classify the dataset. Make the necessary changes to your implementation and show the new decision boundary in your report.

# In[297]:


train = pd.read_csv("datasets/classification/cl_train_2.csv", header=None).values
test = pd.read_csv("datasets/classification/cl_test_2.csv", header=None).values


# In[298]:


x_train, y_train = prep(train)
x_test, y_test = prep(test)
x_train.shape, y_train.shape, x_test.shape, y_test.shape


# In[302]:


tr_errors, test_errors, tr_a, test_a, weights = logreg_train(x_train, y_train, 
                                                  x_test, y_test, lr=0.1)


# In[303]:


plot_errors(tr_errors, test_errors)


# In[304]:


plot_acc(tr_a, test_a)


# In[305]:


plot_lin_db(train, line)


# We can see that this data is _not_ linearly separable. There exists no straight line that is able to separate the two classes in this two-dimensional feature space. To mitigate this, we can add polynomial interactions between the features. This will allow our model fit data that is not only linearly separable, but separable by a function given by the order of the polynomial features we add. 

# In[306]:


def add_poly_features(x, order=2):
    '''
    Add polynomial interactions between the two feature columns x[:,1] and x[:,2].
    This function is only valid for a Nx2 or Nx3 array, where the first column is 
    assumed to be an intercept column if there are 3 columns.
    '''
    # If we have two columns
    if x.shape[1] == 2:
        # Split the two columns, and create intercept column.
        f1, f2 = np.hsplit(x, x.shape[1])
        ic = np.ones_like(f1)
    else:
        # Split the three columns into separate arrays.
        ic, f1, f2 = np.hsplit(x, x.shape[1])
    f = []
    # Add all interaction of the features up to the desired order. 
    for i in range(1, order + 1):
        for j in range(i + 1):
            f.append(np.power(f1.flatten(), i - j) * np.power(f2.flatten(), j))
    # Prepend intercept column
    return np.concatenate([ic, np.array(f).T], axis=1)


# In[314]:


def draw_boundary(weights, order=2):
    '''
    Draw a decision boundary implied by the weights. 
    We do this by creating samples across a 100x100 grid over the feature space, and 
    evaluating our decision function for each sample. We then get the probabilities 
    for each sample in the grid, and can make a contour plot where the probabilities 
    exceeds 0.5.
    '''
    # Create a meshgrid consisting of 2 100x100 arrays with equal steps from 0 to 1. 
    x1, x2 = np.meshgrid(np.linspace(0,1,100), np.linspace(0,1,100))
    # Add polynomial features to our 2d samples.
    x_mesh = add_poly_features(np.stack([x1.flatten(), x2.flatten()]).T)
    # Evaluate all samples over our decision function, and reshape back to 100x100
    # The z will now contain a probability for each of the 10000 points in our grid. 
    z = sigmoid(x_mesh.dot(weights).reshape(x1.shape))
    # Make a contour plot, drawing a line where the probabilities exceed 0.5.
    db = plt.contour(x1, x2, z, levels=[0.5], colors=['g'])


# In[308]:


x_train_poly = add_poly_features(x_train)
x_test_poly = add_poly_features(x_test)
x_train_poly.shape, x_test_poly.shape


# In[316]:


tr_errors, test_errors, tr_a, test_a, weights = logreg_train(x_train_poly, y_train, x_test_poly, y_test,  lr=0.3, iterations=6000)


# In[317]:


plot_errors(tr_errors, test_errors)


# In[318]:


plot_acc(tr_a, test_a)


# In[312]:


weights


# In[315]:


pos = plt.scatter(x_train[np.where(y_train>0),1], x_train[np.where(y_train>0), 2])
neg = plt.scatter(x_train[np.where(y_train==0),1], x_train[np.where(y_train==0), 2])
plt.legend((pos, neg),('Positive', 'Negative'))
plt.title("Logistic regression data and polynomial decision boundary")
draw_boundary(weights)


# We can see that our model trained on input data with added polynomial interactions between the features is able to fit the data very well. 

# In[ ]:




