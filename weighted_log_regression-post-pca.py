import math
import seaborn as sns
%matplotlib inline
from sknn.mlp import Regressor, Layer
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

means = np.reshape(np.apply_along_axis(np.mean,0,X_train),(1,-1))

X_train_norm1 = X_train - means
stds = np.reshape(np.apply_along_axis(np.std,0,X_train_norm1),(1,-1))

X_train_normalized = X_train_norm1/stds


Sigma = X_train_normalized.T.dot(X_train_normalized)/m


w, v = np.linalg.eig(Sigma)



def train_logistic_regression(X_train, Y_train,sample_weights,X_dev):
    model = linear_model.LogisticRegression()
    model.fit(X_train,Y_train,sample_weight = sample_weights)
    
    y_train_hat = model.predict(X_train)
    y_dev_hat = model.predict(X_dev)
    
    return (model,y_train_hat,y_dev_hat)


def normalize(X,means,stds,v,n):
    return ((X-means)/stds).dot(v[:,0:n])

def best_num_pca(v,X_train,X_dev,means,stds,Y_train,Y_dev,outputs):
    best_l = 0
    best_val = 0.
    for l in range(10,30):
        X_train_norm = normalize(X_train,means,stds,v,l)
        X_dev_norm = normalize(X_dev,means,stds,v,l)
        (model,y_train_hat,y_dev_hat) = train_logistic_regression(X_train_norm,np.floor((np.sign(Y_train) + 1)/2),np.abs(Y_train),X_dev_norm)
        perf_l = np.average(y_dev_hat * outputs)
        if perf_l > best_val:
            best_l = l
            best_val= perf_l
            best_model = model
        print(l)
        print(perf_l)
    print(best_l)
    print(best_val)
    print(np.average(outputs))
    return best_l,best_val,best_model
        
        
l5,val,model_5 = best_num_pca(v,X_train,X_dev,means,stds,Y_train[:,0],Y_dev[:,0],outputs[a:b,0])
y_hat_train_5 = model_5.predict(normalize(X_train,means,stds,v,l5))
y_hat_dev_5 = model_5.predict(normalize(X_dev,means,stds,v,l5))
y_hat_test_5 = model_5.predict(normalize(X_test,means,stds,v,l5))


l10,val,model_10 = best_num_pca(v,X_train,X_dev,means,stds,Y_train[:,1],Y_dev[:,1],outputs[a:b,1])
y_hat_train_10 = model_10.predict(normalize(X_train,means,stds,v,l10))
y_hat_dev_10 = model_10.predict(normalize(X_dev,means,stds,v,l10))
y_hat_test_10 = model_10.predict(normalize(X_test,means,stds,v,l10))


l20,val,model_20 = best_num_pca(v,X_train,X_dev,means,stds,Y_train[:,2],Y_dev[:,2],outputs[a:b,2])
y_hat_train_20 = model_20.predict(normalize(X_train,means,stds,v,l20))
y_hat_dev_20 = model_20.predict(normalize(X_dev,means,stds,v,l20))
y_hat_test_20 = model_20.predict(normalize(X_test,means,stds,v,l20))

y_train_hat_5 = y_hat_train_5
y_train_hat_10 = y_hat_train_10
y_train_hat_20 = y_hat_train_20

y_dev_hat_5 = y_hat_dev_5
y_dev_hat_10 = y_hat_dev_10
y_dev_hat_20 = y_hat_dev_20

y_test_hat_5 = y_hat_test_5
y_test_hat_10 = y_hat_test_10
y_test_hat_20 = y_hat_test_20



print('Train Weighted Accuracy (5 min): ',model_5.score(normalize(X_train,means,stds,v,l5),np.floor((np.sign(Y_train[:,0]) + 1)/2),sample_weight = np.abs(Y_train[:,0])))
print('Train Gains (5 min): ', np.average((y_train_hat_5 * outputs[0:a,0])))
print('Train Standardized Gains (5 min): ', np.average(outputs[0:a,0]))
print('Train AUC (5 min): ', metrics.roc_auc_score(np.floor((np.sign(Y_train[:,0]) + 1)/2),model_5.predict_proba(normalize(X_train,means,stds,v,l5))[:,1],sample_weight = np.abs(Y_train[:,0])))

print('Dev Weighted Accuracy (5 min): ',model_5.score(normalize(X_dev,means,stds,v,l5),np.floor((np.sign(Y_dev[:,0]) + 1)/2),sample_weight = np.abs(Y_dev[:,0])))
print('Dev Gains (5 min): ', np.average((y_dev_hat_5 * outputs[a:b,0])))
print('Dev Standardized Gains (5 min): ', np.average(outputs[a:b,0]))
print('Dev AUC (5 min): ', metrics.roc_auc_score(np.floor((np.sign(Y_dev[:,0]) + 1)/2),model_5.predict_proba(normalize(X_dev,means,stds,v,l5))[:,1],sample_weight = np.abs(Y_dev[:,0])))

print('Test Weighted Accuracy (5 min): ',model_5.score(normalize(X_test,means,stds,v,l5),np.floor((np.sign(Y_test[:,0]) + 1)/2),sample_weight = np.abs(Y_test[:,0])))
print('Test Gains (5 min): ', np.average((y_test_hat_5 * outputs[b:,0])))
print('Test Standardized Gains (5 min): ', np.average(outputs[b:,0]))
print('Test AUC (5 min): ', metrics.roc_auc_score(np.floor((np.sign(Y_test[:,0]) + 1)/2),model_5.predict_proba(normalize(X_test,means,stds,v,l5))[:,1],sample_weight = np.abs(Y_test[:,0])))

plt.plot(metrics.roc_curve(np.floor((np.sign(Y_test[:,0]) + 1)/2),model_5.predict_proba(normalize(X_test,means,stds,v,l5))[:,1],sample_weight = np.abs(Y_test[:,0]))[0],metrics.roc_curve(np.floor((np.sign(Y_test[:,0]) + 1)/2),model_5.predict_proba(normalize(X_test,means,stds,v,l5))[:,1],sample_weight = np.abs(Y_test[:,0]))[1])


print('Train Weighted Accuracy (10 min): ',model_10.score(normalize(X_train,means,stds,v,l10),np.floor((np.sign(Y_train[:,1]) + 1)/2),sample_weight = np.abs(Y_train[:,1])))
print('Train Gains (10 min): ', np.average((y_train_hat_10 * outputs[0:a,1])))
print('Train Standardized Gains (10 min): ', np.average(outputs[0:a,1]))
print('Train AUC (10 min): ',metrics.roc_auc_score(np.floor((np.sign(Y_train[:,1]) + 1)/2),model_10.predict_proba(normalize(X_train,means,stds,v,l10))[:,1],sample_weight = np.abs(Y_train[:,1])))

print('Dev Weighted Accuracy (10 min): ',model_10.score(normalize(X_dev,means,stds,v,l10),np.floor((np.sign(Y_dev[:,1]) + 1)/2),sample_weight = np.abs(Y_dev[:,1])))
print('Dev Gains (10 min): ', np.average((y_dev_hat_10 * outputs[a:b,1])))
print('Dev Standardized Gains (10 min): ', np.average(outputs[a:b,1]))
print('Dev AUC (10 min): ',metrics.roc_auc_score(np.floor((np.sign(Y_dev[:,1]) + 1)/2),model_10.predict_proba(normalize(X_dev,means,stds,v,l10))[:,1],sample_weight = np.abs(Y_dev[:,1])))

print('Test Weighted Accuracy (10 min): ',model_10.score(normalize(X_test,means,stds,v,l10),np.floor((np.sign(Y_test[:,1]) + 1)/2),sample_weight = np.abs(Y_test[:,1])))
print('Test Gains (10 min): ', np.average((y_test_hat_10 * outputs[b:,1])))
print('Test Standardized Gains (10 min): ', np.average(outputs[b:,1]))
print('Test AUC (10 min): ',metrics.roc_auc_score(np.floor((np.sign(Y_test[:,1]) + 1)/2),model_10.predict_proba(normalize(X_test,means,stds,v,l10))[:,1],sample_weight = np.abs(Y_test[:,1])))
plt.plot(metrics.roc_curve(np.floor((np.sign(Y_test[:,1]) + 1)/2),model_10.predict_proba(normalize(X_test,means,stds,v,l10))[:,1],sample_weight = np.abs(Y_test[:,1]))[0],metrics.roc_curve(np.floor((np.sign(Y_test[:,1]) + 1)/2),model_10.predict_proba(normalize(X_test,means,stds,v,l10))[:,1],sample_weight = np.abs(Y_test[:,1]))[1])


print('Train Weighted Accuracy (20 min): ',model_20.score(normalize(X_train,means,stds,v,l20),np.floor((np.sign(Y_train[:,2]) + 1)/2),sample_weight = np.abs(Y_train[:,2])))
print('Train Gains (20 min): ', np.average((y_train_hat_20 * outputs[0:a,2])))
print('Train Standardized Gains (20 min): ', np.average(outputs[0:a,2]))
print('Train AUC (20 min): ',metrics.roc_auc_score(np.floor((np.sign(Y_train[:,2]) + 1)/2),model_20.predict_proba(normalize(X_train,means,stds,v,l20))[:,1],sample_weight = np.abs(Y_train[:,2])))

print('Dev Weighted Accuracy (20 min): ',model_20.score(normalize(X_dev,means,stds,v,l20),np.floor((np.sign(Y_dev[:,2]) + 1)/2),sample_weight = np.abs(Y_dev[:,2])))
print('Dev Gains (20 min): ', np.average((y_dev_hat_20 * outputs[a:b,2])))
print('Dev Standardized Gains (20 min): ', np.average(outputs[a:b,2]))
print('Dev AUC (20 min): ',metrics.roc_auc_score(np.floor((np.sign(Y_dev[:,2]) + 1)/2),model_20.predict_proba(normalize(X_dev,means,stds,v,l20))[:,1],sample_weight = np.abs(Y_dev[:,2])))

print('Test Weighted Accuracy (20 min): ',model_20.score(normalize(X_test,means,stds,v,l20),np.floor((np.sign(Y_test[:,2]) + 1)/2),sample_weight = np.abs(Y_test[:,2])))
print('Test Gains (20 min): ', np.average((y_test_hat_20 * outputs[b:,2])))
print('Test Standardized Gains (20 min): ', np.average(outputs[b:,2]))
print('Test AUC (20 min): ',metrics.roc_auc_score(np.floor((np.sign(Y_test[:,2]) + 1)/2),model_20.predict_proba(normalize(X_test,means,stds,v,l20))[:,1],sample_weight = np.abs(Y_test[:,2])))
plt.plot(metrics.roc_curve(np.floor((np.sign(Y_test[:,2]) + 1)/2),model_20.predict_proba(normalize(X_test,means,stds,v,l20))[:,1],sample_weight = np.abs(Y_test[:,2]))[0],metrics.roc_curve(np.floor((np.sign(Y_test[:,2]) + 1)/2),model_20.predict_proba(normalize(X_test,means,stds,v,l20))[:,1],sample_weight = np.abs(Y_test[:,2]))[1])

import pickle
model_means_3 = means
model_stds_3 = stds
model_l5 = l5
model_l10 = l10
model_l20 = l20
pickle.dump(model_means_3,open("model_possibility_pca_5_means.p","wb"))
pickle.dump(model_stds_3,open("model_possibility_pca_5_stds.p","wb"))
pickle.dump(model_l5,open("model_possibility_pca_5_l5.p","wb"))
pickle.dump(model_l10,open("model_possibility_pca_5_l10.p","wb"))
pickle.dump(model_l20,open("model_possibility_pca_5_l20.p","wb"))
model_possibility_3 = model_5
pickle.dump( model_possibility_3, open( "model_possibility_pca_5min_5.p", "wb" ) )
model_possibility_3 = model_10
pickle.dump( model_possibility_3, open( "model_possibility_pca_10min_5.p", "wb" ) )
model_possibility_3 = model_20
pickle.dump( model_possibility_3, open( "model_possibility_pca_20min_5.p", "wb" ) )