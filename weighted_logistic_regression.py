import math
import seaborn as sns
%matplotlib inline
from sknn.mlp import Regressor, Layer
import numpy as np
from numpy import genfromtxt
import math
from sklearn import linear_model
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

temp = np.concatenate((outputs,features), axis=1)
np.random.shuffle(temp)
outputs = temp[:,0:3]
features = temp[:,3:]

m = modoutputs.shape[0]

a = m * 6/10
b = m * 8/10

Y_train = outputs[0:a,:]
X_train = features[0:a,:]
Y_dev = outputs[a:b,:]
X_dev = features[a:b,:]
Y_test = outputs[b:,:]
X_test = features[b:,:]

#mod_wls = sm.WLS(Y_train, X_train)

log_mod_wls_5 = linear_model.LogisticRegression()
log_mod_wls_10 = linear_model.LogisticRegression()
log_mod_wls_20 = linear_model.LogisticRegression()
log_mod_wls_5.fit(X_train,np.floor((np.sign(Y_train[:,0]) + 1)/2),sample_weight = np.abs(Y_train[:,0]))
log_mod_wls_10.fit(X_train,np.floor((np.sign(Y_train[:,1]) + 1)/2),sample_weight = np.abs(Y_train[:,1]))
log_mod_wls_20.fit(X_train,np.floor((np.sign(Y_train[:,2]) + 1)/2),sample_weight = np.abs(Y_train[:,2]))


y_train_hat_5 = log_mod_wls_5.predict(X_train)
y_dev_hat_5 = log_mod_wls_5.predict(X_dev)
y_test_hat_5 = log_mod_wls_5.predict(X_test)

print('Train Cost (5 min): ', np.sum(np.abs(Y_train[:,0]) * ((y_train_hat_5 - Y_train[:,0])**2))/np.sum(np.abs(Y_train[:,0])))
print('Train Accuracy (5 min): ', np.average(y_train_hat_5 == (np.sign(Y_train[:,0]) +1)/2))
print('Train Weighted Accuracy (5 min): ',log_mod_wls_5.score(X_train,np.floor((np.sign(Y_train[:,0]) + 1)/2),sample_weight = np.abs(Y_train[:,0])))
print('Train Gains (5 min): ', np.average((y_train_hat_5 * outputs[0:a,0])))
print('Train Standardized Gains (5 min): ', np.average(outputs[0:a,0]))

print('Dev Cost (5 min): ', np.sum(np.abs(Y_dev[:,0]) * ((y_dev_hat_5 - Y_dev[:,0])**2))/np.sum(np.abs(Y_dev[:,0])))
print('Dev Accuracy (5 min): ', np.average(y_dev_hat_5 == (np.sign(Y_dev[:,0])+1)/2))
print('Dev Weighted Accuracy (5 min): ',log_mod_wls_5.score(X_dev,np.floor((np.sign(Y_dev[:,0]) + 1)/2),sample_weight = np.abs(Y_dev[:,0])))
print('Dev Gains (5 min): ', np.average((y_dev_hat_5 * outputs[a:b,0])))
print('Dev Standardized Gains (5 min): ', np.average(outputs[a:b,0]))

print('Test Cost (5 min): ', np.sum(np.abs(Y_test[:,0]) * ((y_test_hat_5 - Y_test[:,0])**2))/np.sum(np.abs(Y_test[:,0])))
print('Test Accuracy (5 min): ', np.average(y_test_hat_5 == (np.sign(Y_test[:,0])+1)/2))
print('Test Weighted Accuracy (5 min): ',log_mod_wls_5.score(X_test,np.floor((np.sign(Y_test[:,0]) + 1)/2),sample_weight = np.abs(Y_test[:,0])))
print('Test Gains (5 min): ', np.average((y_test_hat_5 * outputs[b:,0])))
print('Test Standardized Gains (5 min): ', np.average(outputs[b:,0]))


y_train_hat_10 = log_mod_wls_10.predict(X_train)
y_dev_hat_10 = log_mod_wls_10.predict(X_dev)
y_test_hat_10 = log_mod_wls_10.predict(X_test)

print('Train Cost (10 min): ', np.sum(np.abs(Y_train[:,1]) * ((y_train_hat_10 - Y_train[:,1])**2))/np.sum(np.abs(Y_train[:,1])))
print('Train Accuracy (10 min): ', np.average(y_train_hat_10 == (np.sign(Y_train[:,1]) +1)/2))
print('Train Weighted Accuracy (10 min): ',log_mod_wls_10.score(X_train,np.floor((np.sign(Y_train[:,1]) + 1)/2),sample_weight = np.abs(Y_train[:,1])))
print('Train Gains (10 min): ', np.average((y_train_hat_10 * outputs[0:a,1])))
print('Train Standardized Gains (10 min): ', np.average(outputs[0:a,1]))

print('Dev Cost (10 min): ', np.sum(np.abs(Y_dev[:,1]) * ((y_dev_hat_10 - Y_dev[:,1])**2))/np.sum(np.abs(Y_dev[:,1])))
print('Dev Accuracy (10 min): ', np.average(y_dev_hat_10 == (np.sign(Y_dev[:,1])+1)/2))
print('Dev Weighted Accuracy (10 min): ',log_mod_wls_10.score(X_dev,np.floor((np.sign(Y_dev[:,1]) + 1)/2),sample_weight = np.abs(Y_dev[:,1])))
print('Dev Gains (10 min): ', np.average((y_dev_hat_10 * outputs[a:b,1])))
print('Dev Standardized Gains (10 min): ', np.average(outputs[a:b,1]))

print('Test Cost (10 min): ', np.sum(np.abs(Y_test[:,1]) * ((y_test_hat_10 - Y_test[:,1])**2))/np.sum(np.abs(Y_test[:,1])))
print('Test Accuracy (10 min): ', np.average(y_test_hat_10 == (np.sign(Y_test[:,1])+1)/2))
print('Test Weighted Accuracy (10 min): ',log_mod_wls_10.score(X_test,np.floor((np.sign(Y_test[:,1]) + 1)/2),sample_weight = np.abs(Y_test[:,1])))
print('Test Gains (10 min): ', np.average((y_test_hat_10 * outputs[b:,1])))
print('Test Standardized Gains (10 min): ', np.average(outputs[b:,1]))
print('Test Cost (20 min): ', np.sum(np.abs(Y_test[:,2]) * ((y_test_hat_20 - Y_test[:,2])**2))/np.sum(np.abs(Y_test[:,2])))


y_train_hat_20 = log_mod_wls_20.predict(X_train)
y_dev_hat_20 = log_mod_wls_20.predict(X_dev)
y_test_hat_20 = log_mod_wls_20.predict(X_test)

print('Train Cost (20 min): ', np.sum(np.abs(Y_train[:,2]) * ((y_train_hat_20 - Y_train[:,2])**2))/np.sum(np.abs(Y_train[:,2])))
print('Train Accuracy (20 min): ', np.average(y_train_hat_20 == (np.sign(Y_train[:,2]) +1)/2))
print('Train Weighted Accuracy (20 min): ',log_mod_wls_20.score(X_train,np.floor((np.sign(Y_train[:,2]) + 1)/2),sample_weight = np.abs(Y_train[:,2])))
print('Train Gains (20 min): ', np.average((y_train_hat_20 * outputs[0:a,2])))
print('Train Standardized Gains (20 min): ', np.average(outputs[0:a,2]))

print('Dev Cost (20 min): ', np.sum(np.abs(Y_dev[:,2]) * ((y_dev_hat_20 - Y_dev[:,2])**2))/np.sum(np.abs(Y_dev[:,2])))
print('Dev Accuracy (20 min): ', np.average(y_dev_hat_20 == (np.sign(Y_dev[:,2])+1)/2))
print('Dev Weighted Accuracy (20 min): ',log_mod_wls_20.score(X_dev,np.floor((np.sign(Y_dev[:,2]) + 1)/2),sample_weight = np.abs(Y_dev[:,2])))
print('Dev Gains (20 min): ', np.average((y_dev_hat_20 * outputs[a:b,2])))
print('Dev Standardized Gains (20 min): ', np.average(outputs[a:b,2]))

print('Test Cost (20 min): ', np.sum(np.abs(Y_test[:,2]) * ((y_test_hat_20 - Y_test[:,2])**2))/np.sum(np.abs(Y_test[:,2])))
print('Test Accuracy (20 min): ', np.average(y_test_hat_20 == (np.sign(Y_test[:,2])+1)/2))
print('Test Weighted Accuracy (20 min): ',log_mod_wls_20.score(X_test,np.floor((np.sign(Y_test[:,2]) + 1)/2),sample_weight = np.abs(Y_test[:,2])))
print('Test Gains (20 min): ', np.average((y_test_hat_20 * outputs[b:,2])))
print('Test Standardized Gains (20 min): ', np.average(outputs[b:,2]))




import pickle
model_possibility_1 = log_mod_wls_5
pickle.dump( model_possibility_1, open( "model_possibility_5min_2.p", "wb" ) )
model_possibility_1 = log_mod_wls_10
pickle.dump( model_possibility_1, open( "model_possibility_10min_2.p", "wb" ) )
model_possibility_1 = log_mod_wls_20
pickle.dump( model_possibility_1, open( "model_possibility_20min_2.p", "wb" ) )

