import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torchinfo 
from torch.utils.data import DataLoader

from math import *

device = "cuda" if torch.cuda.is_available() else "cpu"

def formula1(inputSize, outputSize):
    hiddenSize = ceil(2/3 * inputSize + outputSize)
    return hiddenSize

def formula2(sampleSize, inputSize, outputSize):
    alpha = 2
    hiddenSize = ceil(sampleSize/(alpha * (inputSize + outputSize)))
    return hiddenSize

def formula3(numberOfHiddenLayers, layerNum, inputSize, outputSize):
    stepNeurons = round((outputSize - inputSize)/(numberOfHiddenLayers + 1))
    hiddenSize = stepNeurons * layerNum + inputSize
    return hiddenSize

def MSE_loss(y_pred, y_true):
    loss = F.mse_loss
    lossValue = loss(torch.FloatTensor(y_pred), torch.FloatTensor(y_true))
    return lossValue.detach().item()

class NeuralNetwork(nn.Module):
    def __init__(self, inputSize, outputSize, formula, numberOfHiddenLayers, sampleSize=None):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()

        if numberOfHiddenLayers == 1:
            if formula == "formula1":
                hiddenSize = formula1(inputSize, outputSize)
            elif formula == "formula2":
                hiddenSize = formula2(sampleSize, inputSize, outputSize)
            elif formula == "formula3":
                hiddenSize = formula3(numberOfHiddenLayers, 1, inputSize, outputSize)
            
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(inputSize, hiddenSize),
                #nn.Dropout(p=0.2),
                nn.ReLU(),
                nn.Linear(hiddenSize, outputSize),
            )

        elif numberOfHiddenLayers == 2:
            if formula == "formula1":
                hiddenSize1 = formula1(inputSize, outputSize)
                hiddenSize2 = formula1(inputSize, outputSize)
            elif formula == "formula2":
                hiddenSize1 = formula2(sampleSize, inputSize, outputSize)
                hiddenSize2 = formula2(sampleSize, inputSize, outputSize)
            elif formula == "formula3":
                hiddenSize1 = formula3(numberOfHiddenLayers, 1, inputSize, outputSize)
                hiddenSize2 = formula3(numberOfHiddenLayers, 2, inputSize, outputSize)

            self.linear_relu_stack = nn.Sequential(
                nn.Linear(inputSize, hiddenSize1),
                #nn.Dropout(p=0.05),
                nn.ReLU(),
                nn.Linear(hiddenSize1, hiddenSize2),
                #nn.Dropout(p=0.05),
                nn.ReLU(),
                nn.Linear(hiddenSize2, outputSize),
            )

    def forward(self, X):
        X = self.flatten(X)
        output = self.linear_relu_stack(X)
        return output

    # The training loop
    def train(self, X, Y, ANNOptimizer, learning_rate, epochs, L2_regularization):
        losses = []
        X = torch.FloatTensor(X)
        Y = torch.FloatTensor(Y)
        self.zero_grad()
        if ANNOptimizer == "Adam":
            optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=L2_regularization)
        elif ANNOptimizer == "SGD":
            optimizer = optim.SGD(self.parameters(), lr=learning_rate)
        mse_loss = F.mse_loss
        lossValue = 100000
        for i in range(0, epochs):
        #while lossValue > 500:
            optimizer.zero_grad() 
            Y_pred = self(X) # x is the features
            loss = mse_loss(Y_pred, Y)
            lossValue = loss.detach().item()
            #print(lossValue)
            losses.append(lossValue)
            loss.backward()
            optimizer.step()
        return losses

    def predict(self,X):
        tensorX = torch.FloatTensor(X)#.reshape([1,-1])
        #print(tensorX.shape)
        Y_pred = self(tensorX).detach().numpy() 
        return Y_pred

    def predictOneDimension(self,X):
        tensorX = torch.FloatTensor(X).reshape([1,-1])
        #print(tensorX.shape)
        Y_pred = self(tensorX).detach().numpy() 
        return Y_pred
    
    def logModel(self):
        return str(torchinfo.summary(self, batch_dim = 0,verbose = 1))

def logInfo():
    
    messages = [f"PyTorch version: {torch.__version__}\n",
                f"Device used: {device}\n"
                ]
    return messages


# python3 optimizers/ANN.py