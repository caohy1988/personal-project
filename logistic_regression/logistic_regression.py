#!/usr/bin/env python
# encoding: utf-8


'''
from y^hat = exp(b_0 + b_1 * x1) / (1 + exp(b_0 + b_1*x1))
stochastic gradient descent
b = b + learning_rate * error * g'(x)  g(x) = 1 / ( 1 + exp-(b_0 + b_1 * x1))
b = b + learning_rate * (y-y^hat) * y^hat * (1-y_hat) * x

'''
from math import exp

def prediction(row, coefficients):
    yhat = coefficients[0]
    for i in range(len(row)-1):
        yhat += coefficients[i + 1] * row[i]
    return 1.0 / (1.0 + exp(-yhat))

# Estimate logistic regression coeeficients using stochastic gradient descent
def coefficients_sgd(train, learning_rate, n_epoch):
    coef = [0.0 for i in range(len(train[0]))]
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            yhat =prediction(row, coef)
            error = row[-1] - yhat
            sum_error += error**2
            coef[0] = coef[0] + learning_rate * error * yhat * (1.0 - yhat)
            for i in range(len(row)-1):
                coef[i+1] = coef[i+1] + learning_rate* error * yhat * (1.0-yhat) * row[i]
        print ('>epoch=%d, learning_rate=%.3f, error=%.3f'%(epoch, learning_rate, sum_error))
    return coef



dataset = [[2.7810836,2.550537003,0],
[1.465489372,2.362125076,0],
[3.396561688,4.400293529,0],
[1.38807019,1.850220317,0],
[3.06407232,3.005305973,0],
[7.627531214,2.759262235,1],
[5.332441248,2.088626775,1],
[6.922596716,1.77106367,1],
[8.675418651,-0.242068655,1],
[7.673756466,3.508563011,1]]
coef = [-0.406605464, 0.852573316, -1.104746259]
learning_rate = 0.3
n_epoch = 100
coef = coefficients_sgd(dataset, learning_rate, n_epoch)
print (coef)
for row in dataset:
    yhat = prediction(row, coef)
    print("Expected=%.3f, Predicted=%.3f [%d]" % (row[-1], yhat, round(yhat)))