from sknn.mlp import Regressor, Layer
import numpy as np
from numpy import genfromtxt

featuresFile = ''
outputsFile = ''

features= genfromtxt(featuresFile, delimiter=',')
outputs = genfromtxt(outputsFile, delimiter=',')

X_train = features[1:40000]
y_train = outputs[1:40000]
X_dev = features[40000:]
y_dev = outputs[40000:]

nn = Regressor(
    layers=[
        Layer("Rectifier", units=100),
        Layer("Sigmoid")],
    learning_rate=0.02,
    n_iter=10)
nn.fit(X_train, y_train)

y_hat = nn.predict(X_dev)
