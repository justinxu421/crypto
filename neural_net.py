import math
import seaborn as sns
%matplotlib inline
#from sknn.mlp import Regressor, Layer
import numpy as np
from numpy import genfromtxt
import math
from sklearn import linear_model, metrics
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from statsmodels.iolib.table import (SimpleTable, default_txt_fmt)


featuresFile = 'NN_data_pv_raw.csv'
outputsFile = 'NN_data_rv_raw.csv'

#get data
features= genfromtxt(featuresFile, delimiter=',')
outputs = genfromtxt(outputsFile, delimiter=',')

#remove header
features = features[1:]
outputs = outputs[1:]

features[:,0] = np.minimum(features[:,0],1.01)
features[:,1] = np.minimum(features[:,1],1.015)
features[:,2] = np.minimum(features[:,2],1.025)
features[:,3] = np.minimum(features[:,3],1.035)
features[:,4] = np.minimum(features[:,4],1.05)
features[:,5] = np.minimum(features[:,5],1.07)
features[:,6] = np.minimum(features[:,6],1.1)
features[:,7] = np.maximum(features[:,7],0.985)
features[:,8] = np.maximum(features[:,8],0.98)
features[:,9] = np.maximum(features[:,9],0.975)
features[:,10] = np.maximum(features[:,10],0.965)
features[:,11] = np.maximum(features[:,11],0.95)
features[:,12] = np.maximum(features[:,12],0.93)
features[:,13] = np.maximum(features[:,13],0.88)
features[:,14] = np.maximum(np.minimum(features[:,14],1.02),0.98)
features[:,15] = np.maximum(np.minimum(features[:,15],1.02),0.98)
features[:,16] = np.maximum(np.minimum(features[:,16],1.03),0.97)
features[:,17] = np.maximum(np.minimum(features[:,17],1.05),0.95)
features[:,18] = np.maximum(np.minimum(features[:,18],1.07),0.93)
features[:,19] = np.minimum(features[:,19],200)
features[:,20] = np.minimum(features[:,20],400)
features[:,21] = np.minimum(features[:,21],600)
features[:,22] = np.minimum(features[:,22],1000)
features[:,23] = np.minimum(features[:,23],2000)
features[:,24] = np.minimum(features[:,24],3500)
features[:,25] = np.minimum(features[:,25],6000)
features[:,26] = np.maximum(np.minimum(features[:,26],1.015),0.985)
features[:,27] = np.maximum(np.minimum(features[:,27],1.02),0.98)
features[:,28] = np.maximum(np.minimum(features[:,28],1.03),0.97)
features[:,29] = np.maximum(np.minimum(features[:,29],1.04),0.96)
features[:,30] = np.maximum(np.minimum(features[:,30],1.05),0.95)
features[:,31] = np.maximum(np.minimum(features[:,31],1.07),0.93)
features[:,32] = np.maximum(np.minimum(features[:,32],1.1),0.9)
features[:,40] = np.maximum(np.minimum(features[:,40],1.015),0.985)
features[:,41] = np.maximum(np.minimum(features[:,41],1.015),0.985)
features[:,42] = np.maximum(np.minimum(features[:,42],1.03),0.97)
features[:,43] = np.maximum(np.minimum(features[:,43],1.03),0.97)
features[:,44] = np.maximum(np.minimum(features[:,44],1.04),0.96)
features[:,45] = np.maximum(np.minimum(features[:,45],1.05),0.95)
features[:,46] = np.maximum(np.minimum(features[:,46],1.015),0.985)
features[:,47] = np.maximum(np.minimum(features[:,47],1.02),0.98)
features[:,48] = np.maximum(np.minimum(features[:,48],1.04),0.96)
features[:,49] = np.maximum(np.minimum(features[:,49],1.04),0.96)
features[:,50] = np.maximum(np.minimum(features[:,50],1.06),0.94)
features[:,51] = np.maximum(np.minimum(features[:,51],1.08),0.92)
features[:,59] = np.log(features[:,59] + 0.0001)
features[:,60] = np.log(features[:,60] + 0.0001)
features[:,61] = np.log(features[:,61] + 0.0001)
features[:,62] = np.log(features[:,62] + 0.0001)
features[:,63] = np.log(features[:,63] + 0.0001)
features[:,64] = np.log(features[:,64] + 0.0001)
features[:,65] = np.log(features[:,65] + 0.0001)
#outputs[:,0] = abs(outputs[:,0])**(0.5) * (np.sign(outputs[:,0]))
#outputs[:,1] = abs(outputs[:,1])**(0.6) * (np.sign(outputs[:,1]))
#outputs[:,2] = abs(outputs[:,2])**(0.7) * (np.sign(outputs[:,2]))

col_idx = np.concatenate((range(0,26),range(33,40),range(46,59)),axis=0)
#remove R, WAP, VR
features = features[:,col_idx]

outputs = np.log(1 + outputs)


m = outputs.shape[0]

a = m * 6/10
b = m * 8/10

Y_train = outputs[0:a,:]
X_train = features[0:a,:]
Y_dev = outputs[a:b,:]
X_dev = features[a:b,:]
Y_test = outputs[b:,:]
X_test = features[b:,:]


def sigmoid(x):
    m = np.min(x)
    s = np.exp(m)/(np.exp(m) + np.exp(m-x))
    return s


def ReLU(x):
    return np.maximum(0,x)

def forward_prop(data, y, weights, params):
    
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    #number of examples
    m = np.shape(data)[0]
    
    #apply first layer (ReLU)
    Z1 = data.dot(W1) + b1.T
    A1 = ReLU(Z1)
    
    #apply second layer (sigmoid)
    Z2 = A1.dot(W2) + b2.T
    y_hat = sigmoid(Z2)
    
    #calculate cost
    cost = -np.sum(weights * ((y * np.log(y_hat)) + (1-y) * np.log(1-y_hat)))
    
    ### END YOUR CODE
    return A1, y_hat, cost


def backward_prop(data, y, weights, params, reg_rate):
    """
    return gradient of parameters
    """
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']
    
    m = data.shape[0]
    y = np.reshape(y,(-1,1))

    h, yhat, cost = forward_prop(data, y, weights, params)
    
    #backpropogation for delta2
    delta2 = np.reshape(weights,(-1,1)) * (yhat - y)
    
    #backpropogation for delta1
    g_term = np.floor((np.sign(h) + 1)/2)
    delta1 = delta2.dot(W2.T) * g_term
    
    #backpropogation for all other terms
    gradW2 = h.T.dot(delta2)/m + reg_rate * 2 * W2
    gradb2 = np.reshape(np.apply_along_axis(np.mean,0,delta2),(1,1))
    gradW1 = data.T.dot(delta1)/m + reg_rate * 2 * W1
    gradb1 = np.reshape(np.apply_along_axis(np.mean,0,delta1),params['b1'].shape)
    ### END YOUR CODE

    grad = {}
    grad['W1'] = gradW1
    grad['W2'] = gradW2
    grad['b1'] = gradb1
    grad['b2'] = gradb2

    return grad
    
    
def nn_train(trainData, train_y, devData, dev_y,learning_rate,reg_rate):
    (m, l) = trainData.shape
    num_hidden = 20
    params = {}

    ### YOUR CODE HERE
    B = m/60
    params = {}
    params['W1'] = np.random.standard_normal((l,num_hidden))
    params['W2'] = np.random.standard_normal((num_hidden,1))
    params['b1'] = np.zeros((num_hidden,1))
    params['b2'] = np.zeros((1,1))
    
    num_epochs = 30
    
    traincost = np.zeros(num_epochs)
    devcost = np.zeros(num_epochs)
    
    trainacc = np.zeros(num_epochs)
    devacc = np.zeros(num_epochs)
    
    trainperf = np.zeros(num_epochs)
    devperf = np.zeros(num_epochs)
    
    for e in range(num_epochs):
        for i in range(60):
            grad = backward_prop(trainData[B*(i):B*(i+1),:], np.floor((np.sign(train_y[B*(i):B*(i+1)]) + 1)/2), np.abs(train_y[B*(i):B*(i+1)]), params,reg_rate)

            params['W1'] = params['W1'] - learning_rate * grad['W1']
            params['W2'] = params['W2'] - learning_rate * grad['W2']
            params['b1'] = params['b1'] - learning_rate * grad['b1']
            params['b2'] = params['b2'] - learning_rate * grad['b2']

        Z1 = trainData.dot(params['W1']) + params['b1'].T
        A1 = ReLU(Z1)
        Z2 = A1.dot(params['W2']) + params['b2'].T
        yhat = sigmoid(Z2)
        
        cost = ((train_y * np.log(yhat)) + (1-train_y) * np.log(1-yhat))
        traincost[e] = -np.sum((np.abs(train_y)) * cost)/np.sum(np.abs(train_y))
        
        trainacc[e] = np.sum(np.abs(train_y) * (np.round(yhat) == np.floor((np.sign(train_y) + 1)/2)))/np.sum(np.abs(train_y))
        
        trainperf[e] = np.average(np.round(yhat) * train_y)
        
        Z1 = devData.dot(params['W1']) + params['b1'].T
        A1 = ReLU(Z1)
        Z2 = A1.dot(params['W2']) + params['b2'].T
        yhat = sigmoid(Z2)
        
        cost = ((dev_y * np.log(yhat)) + (1-dev_y) * np.log(1-yhat))
        devcost[e] = -np.sum((np.abs(dev_y)) * cost)/np.sum(np.abs(dev_y))
        
        devpred = np.round(yhat)
        devacc[e] = np.sum(np.abs(dev_y) * (1-np.abs(devpred - np.floor((np.sign(dev_y) + 1)/2))))/np.sum(np.abs(dev_y))
        
        devperf[e] = np.average(np.round(yhat) * dev_y)
        
        print('epoch: ')
        print(e)
        print('traincost:')
        print(traincost[e])
        print('trainacc:')
        print(trainacc[e])
        print('Training performance: ')
        print(trainperf[e])
        print('Training benchmark: ')
        print(np.average(train_y))
        print('devcost:')
        print(devcost[e])
        print('devacc:')
        print(devacc[e])
        print('Dev performance: ')
        print(devperf[e])
        print('Dev benchmark: ')
        print(np.average(dev_y))

    return params

means = np.reshape(np.apply_along_axis(np.mean,0,X_train),(1,-1))

X_train_norm1 = X_train - means
stds = np.reshape(np.apply_along_axis(np.std,0,X_train_norm1),(1,-1))

X_train_norm = (X_train - means)/stds
X_dev_norm = (X_dev - means)/stds
X_test_norm = (X_test - means)/stds

params5 = nn_train(X_train_norm,np.reshape(Y_train[:,0],(-1,1)),X_dev_norm,np.reshape(Y_dev[:,0],(-1,1)),220,0.000005)

W1 = params5['W1']
b1 = params5['b1']
W2 = params5['W2']
b2 = params5['b2']
    
#apply first layer (ReLU)
Z1 = X_train_norm.dot(W1) + b1.T
A1 = ReLU(Z1)
    
#apply second layer (sigmoid)
Z2 = A1.dot(W2) + b2.T
y_train_hat_5 = sigmoid(Z2)
    
#apply first layer (ReLU)
Z1 = X_dev_norm.dot(W1) + b1.T
A1 = ReLU(Z1)
    
#apply second layer (sigmoid)
Z2 = A1.dot(W2) + b2.T
y_dev_hat_5 = sigmoid(Z2)
    
#apply first layer (ReLU)
Z1 = X_test_norm.dot(W1) + b1.T
A1 = ReLU(Z1)
    
#apply second layer (sigmoid)
Z2 = A1.dot(W2) + b2.T
y_test_hat_5 = sigmoid(Z2)

print('Train Weighted Accuracy (5 min): ',np.sum((np.round(y_train_hat_5) == (np.sign(np.reshape(Y_train[:,0],(-1,1)))+1)/2) * (np.abs(np.reshape(Y_train[:,0],(-1,1)))))/np.sum((np.abs(np.reshape(Y_train[:,0],(-1,1))))))
print('Train Gains (5 min): ', np.average((np.round(np.reshape(y_train_hat_5,(-1,1))) * np.reshape(outputs[0:a,0],(-1,1)))))
print('Train Standardized Gains (5 min): ', np.average(outputs[0:a,0]))
print('Train AUC (5 min): ', metrics.roc_auc_score(np.floor((np.sign(Y_train[:,0]) + 1)/2),y_train_hat_5,sample_weight = np.abs(Y_train[:,0])))

print('Dev Weighted Accuracy (5 min): ',np.sum((np.round(y_dev_hat_5) == (np.sign(np.reshape(Y_dev[:,0],(-1,1)))+1)/2) * (np.abs(np.reshape(Y_dev[:,0],(-1,1)))))/np.sum((np.abs(np.reshape(Y_test[:,0],(-1,1))))))
print('Dev Gains (5 min): ', np.average((np.round(y_dev_hat_5) * np.reshape(outputs[a:b,0],(-1,1)))))
print('Dev Standardized Gains (5 min): ', np.average(outputs[a:b,0]))
print('Dev AUC (5 min): ', metrics.roc_auc_score(np.floor((np.sign(Y_dev[:,0]) + 1)/2),y_dev_hat_5,sample_weight = np.abs(Y_dev[:,0])))

print('Test Weighted Accuracy (5 min): ',np.sum((np.round(y_test_hat_5) == (np.sign(np.reshape(Y_test[:,0],(-1,1)))+1)/2) * (np.abs(np.reshape(Y_test[:,0],(-1,1)))))/np.sum((np.abs(np.reshape(Y_test[:,0],(-1,1))))))
print('Test Gains (5 min): ', np.average((np.round(y_test_hat_5) * np.reshape(outputs[b:,0],(-1,1)))))
print('Test Standardized Gains (5 min): ', np.average(outputs[b:,0]))
print('Test AUC (5 min): ', metrics.roc_auc_score(np.floor((np.sign(Y_test[:,0]) + 1)/2),y_test_hat_5,sample_weight = np.abs(Y_test[:,0])))

plt.plot(metrics.roc_curve(np.floor((np.sign(Y_test[:,0]) + 1)/2),y_test_hat_5,sample_weight = np.abs(Y_test[:,0]))[0],metrics.roc_curve(np.floor((np.sign(Y_test[:,0]) + 1)/2),y_test_hat_5,sample_weight = np.abs(Y_test[:,0]))[1])


params10 = nn_train(X_train_norm,np.reshape(Y_train[:,1],(-1,1)),X_dev_norm,np.reshape(Y_dev[:,1],(-1,1)),200,0.0000025)

W1 = params10['W1']
b1 = params10['b1']
W2 = params10['W2']
b2 = params10['b2']
    
#apply first layer (ReLU)
Z1 = X_train_norm.dot(W1) + b1.T
A1 = ReLU(Z1)
    
#apply second layer (sigmoid)
Z2 = A1.dot(W2) + b2.T
y_train_hat_10 = sigmoid(Z2)
    
#apply first layer (ReLU)
Z1 = X_dev_norm.dot(W1) + b1.T
A1 = ReLU(Z1)
    
#apply second layer (sigmoid)
Z2 = A1.dot(W2) + b2.T
y_dev_hat_10 = sigmoid(Z2)
    
#apply first layer (ReLU)
Z1 = X_test_norm.dot(W1) + b1.T
A1 = ReLU(Z1)
    
#apply second layer (sigmoid)
Z2 = A1.dot(W2) + b2.T
y_test_hat_10 = sigmoid(Z2)

print('Train Weighted Accuracy (10 min): ',np.sum((np.round(y_train_hat_10) == (np.sign(np.reshape(Y_train[:,1],(-1,1)))+1)/2) * (np.abs(np.reshape(Y_train[:,1],(-1,1)))))/np.sum((np.abs(np.reshape(Y_train[:,1],(-1,1))))))
print('Train Gains (10 min): ', np.average((np.round(np.reshape(y_train_hat_10,(-1,1))) * np.reshape(outputs[0:a,1],(-1,1)))))
print('Train Standardized Gains (10 min): ', np.average(outputs[0:a,1]))
print('Train AUC (10 min): ', metrics.roc_auc_score(np.floor((np.sign(Y_train[:,1]) + 1)/2),y_train_hat_10,sample_weight = np.abs(Y_train[:,1])))

print('Dev Weighted Accuracy (10 min): ',np.sum((np.round(y_dev_hat_10) == (np.sign(np.reshape(Y_dev[:,1],(-1,1)))+1)/2) * (np.abs(np.reshape(Y_dev[:,1],(-1,1)))))/np.sum((np.abs(np.reshape(Y_test[:,1],(-1,1))))))
print('Dev Gains (10 min): ', np.average((np.round(y_dev_hat_10) * np.reshape(outputs[a:b,1],(-1,1)))))
print('Dev Standardized Gains (10 min): ', np.average(outputs[a:b,1]))
print('Dev AUC (10 min): ', metrics.roc_auc_score(np.floor((np.sign(Y_dev[:,1]) + 1)/2),y_dev_hat_10,sample_weight = np.abs(Y_dev[:,1])))

print('Test Weighted Accuracy (10 min): ', np.sum((np.round(y_test_hat_10) == (np.sign(np.reshape(Y_test[:,1],(-1,1)))+1)/2) * (np.abs(np.reshape(Y_test[:,1],(-1,1)))))/np.sum((np.abs(np.reshape(Y_test[:,1],(-1,1))))))
print('Test Gains (10 min): ', np.average((np.round(y_test_hat_10) * np.reshape(outputs[b:,1],(-1,1)))))
print('Test Standardized Gains (10 min): ', np.average(outputs[b:,1]))
print('Test AUC (10 min): ', metrics.roc_auc_score(np.floor((np.sign(Y_test[:,1]) + 1)/2),y_test_hat_10,sample_weight = np.abs(Y_test[:,1])))

plt.plot(metrics.roc_curve(np.floor((np.sign(Y_test[:,1]) + 1)/2),y_test_hat_10,sample_weight = np.abs(Y_test[:,1]))[0],metrics.roc_curve(np.floor((np.sign(Y_test[:,1]) + 1)/2),y_test_hat_10,sample_weight = np.abs(Y_test[:,1]))[1])


params20 = nn_train(X_train_norm,np.reshape(Y_train[:,2],(-1,1)),X_dev_norm,np.reshape(Y_dev[:,2],(-1,1)),200,0.000005)

W1 = params20['W1']
b1 = params20['b1']
W2 = params20['W2']
b2 = params20['b2']
    
#apply first layer (ReLU)
Z1 = X_train_norm.dot(W1) + b1.T
A1 = ReLU(Z1)
    
#apply second layer (sigmoid)
Z2 = A1.dot(W2) + b2.T
y_train_hat_20 = sigmoid(Z2)
    
#apply first layer (ReLU)
Z1 = X_dev_norm.dot(W1) + b1.T
A1 = ReLU(Z1)
    
#apply second layer (sigmoid)
Z2 = A1.dot(W2) + b2.T
y_dev_hat_20 = sigmoid(Z2)
    
#apply first layer (ReLU)
Z1 = X_test_norm.dot(W1) + b1.T
A1 = ReLU(Z1)
    
#apply second layer (sigmoid)
Z2 = A1.dot(W2) + b2.T
y_test_hat_20 = sigmoid(Z2)

print('Train Weighted Accuracy (20 min): ',np.sum((np.round(y_train_hat_20) == (np.sign(np.reshape(Y_train[:,2],(-1,1)))+1)/2) * (np.abs(np.reshape(Y_train[:,2],(-1,1)))))/np.sum((np.abs(np.reshape(Y_train[:,2],(-1,1))))))
print('Train Gains (20 min): ', np.average((np.round(np.reshape(y_train_hat_20,(-1,1))) * np.reshape(outputs[0:a,2],(-1,1)))))
print('Train Standardized Gains (20 min): ', np.average(outputs[0:a,2]))
print('Train AUC (20 min): ', metrics.roc_auc_score(np.floor((np.sign(Y_train[:,2]) + 1)/2),y_train_hat_20,sample_weight = np.abs(Y_train[:,2])))

print('Dev Weighted Accuracy (20 min): ',np.sum((np.round(y_dev_hat_20) == (np.sign(np.reshape(Y_dev[:,2],(-1,1)))+1)/2) * (np.abs(np.reshape(Y_dev[:,2],(-1,1)))))/np.sum((np.abs(np.reshape(Y_test[:,2],(-1,1))))))
print('Dev Gains (20 min): ', np.average((np.round(y_dev_hat_20) * np.reshape(outputs[a:b,2],(-1,1)))))
print('Dev Standardized Gains (20 min): ', np.average(outputs[a:b,2]))
print('Dev AUC (20 min): ', metrics.roc_auc_score(np.floor((np.sign(Y_dev[:,2]) + 1)/2),y_dev_hat_20,sample_weight = np.abs(Y_dev[:,2])))

print('Test Weighted Accuracy (20 min): ',np.sum((np.round(y_test_hat_20) == (np.sign(np.reshape(Y_test[:,2],(-1,1)))+1)/2) * (np.abs(np.reshape(Y_test[:,2],(-1,1)))))/np.sum((np.abs(np.reshape(Y_test[:,2],(-1,1))))))
print('Test Gains (20 min): ', np.average((np.round(y_test_hat_20) * np.reshape(outputs[b:,2],(-1,1)))))
print('Test Standardized Gains (20 min): ', np.average(outputs[b:,2]))
print('Test AUC (20 min): ', metrics.roc_auc_score(np.floor((np.sign(Y_test[:,2]) + 1)/2),y_test_hat_20,sample_weight = np.abs(Y_test[:,2])))

plt.plot(metrics.roc_curve(np.floor((np.sign(Y_test[:,2]) + 1)/2),y_test_hat_20,sample_weight = np.abs(Y_test[:,2]))[0],metrics.roc_curve(np.floor((np.sign(Y_test[:,2]) + 1)/2),y_test_hat_20,sample_weight = np.abs(Y_test[:,2]))[1])


import pickle
model_means_8 = means
model_stds_8 = stds
pickle.dump(model_means_8,open("model_possibility_nn_8_means.p","wb"))
pickle.dump(model_stds_8,open("model_possibility_nn_8_stds.p","wb"))
model_possibility_3 = params5
pickle.dump( model_possibility_3, open( "model_possibility_nn_5min_8.p", "wb" ) )
model_possibility_3 = params10
pickle.dump( model_possibility_3, open( "model_possibility_nn_10min_8.p", "wb" ) )
model_possibility_3 = params20
pickle.dump( model_possibility_3, open( "model_possibility_nn_20min_8.p", "wb" ) )