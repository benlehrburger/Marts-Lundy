import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import random_split
from numpy import vstack
from numpy import argmax
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from  sklearn.metrics import accuracy_score
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Softmax
from torch.nn import Module
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_
from sklearn.preprocessing import LabelEncoder
import xlrd
import numpy as np

#
#
# PUT TRAIN AND TEST DATA IN ARRAY
#
#

loc = 'dataset.xls'
wb = xlrd.open_workbook(loc)
train_data_file = wb.sheet_by_index(0)
train_labels_file = wb.sheet_by_index(1)
test_data_file = wb.sheet_by_index(2)
test_labels_file = wb.sheet_by_index(3)
total_labels_file = wb.sheet_by_index(4)

train_data = []
train_labels = []
test_data = []
test_labels = []
total_labels = []

# Training data
for row in range(1, train_data_file.nrows):
    features = []
    for col in range(1, train_data_file.ncols):
        features.append(train_data_file.cell_value(row, col))
    train_data.append(features)
train_data = np.array(train_data)

# Training labels
for row in range(1, train_labels_file.nrows):
    features = train_labels_file.cell_value(row, 1)
    train_labels.append(features)
train_labels = np.array(train_labels)

# Testing data
for row in range(1, test_data_file.nrows):
    features = []
    for col in range(1, test_data_file.ncols):
        features.append(test_data_file.cell_value(row, col))
    test_data.append(features)
test_data = np.array(test_data)

# Testing labels
for row in range(1, test_labels_file.nrows):
    features = [test_labels_file.cell_value(row, 1)]
    test_labels.append(features)
test_labels = np.array(test_labels)

# Total labels
for row in range(1, total_labels_file.nrows):
    features = [total_labels_file.cell_value(row, 1)]
    total_labels.append(features)
total_labels = np.array(total_labels)

#
#
# HANDLE DATA
#
#

class Data(Dataset):
    # load the dataset
    def __init__(self, path):
        # store the inputs and outputs
        self.X = train_data
        self.y = total_labels
        # ensure input data is floats
        self.X = self.X.astype('float32')

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

    # get indexes for train and test rows
    def get_splits(self, n_test=0.33):
        # determine sizes
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        # calculate the split
        return random_split(self, [train_size, test_size])

#
#
# PUT TRAIN AND TEST DATA IN ARRAY
#
#

class MLP(Module):
    # define model elements
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        # input to first hidden layer
        self.hidden1 = Linear(n_inputs, 64)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()
        # second hidden layer
        self.hidden2 = Linear(64, 128)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()
        # third hidden layer and output
        self.hidden3 = Linear(128, 12)
        xavier_uniform_(self.hidden3.weight)
        self.act3 = Softmax(dim=1)

    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
        # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # output layer
        X = self.hidden3(X)
        X = self.act3(X)
        return X

# prepare the dataset
def prepare_data(path):
    # prepare data loaders
    return None

# train the model
def train_model(train_dl, model):
    # define the optimization
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    # enumerate epochs
    for epoch in range(500):
        # enumerate mini batches
        for i, (inputs, targets) in enumerate(train_dl):
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = model(inputs)
            # calculate loss
            print(targets.shape)
            loss = criterion(yhat, targets.flatten().type(torch.LongTensor))
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()


# evaluate the model
def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        # evaluate the model on the test set
        yhat = model(inputs)
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        # convert to class labels
        yhat = argmax(yhat, axis=1)
        # reshape for stacking
        actual = actual.reshape((len(actual), 1))
        yhat = yhat.reshape((len(yhat), 1))
        # store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate accuracy
    acc = accuracy_score(actuals, predictions)
    return acc


# make a class prediction for one row of data
def predict(row, model):
    # convert row to data
    row = Tensor([row])
    # make prediction
    yhat = model(row)
    # retrieve numpy array
    yhat = yhat.detach().numpy()
    return yhat


# prepare the data
train_dl = DataLoader(Data(train_data), batch_size=32, shuffle=True)
test_dl = DataLoader(Data(test_data), batch_size=1024, shuffle=False)
# define the network
model = MLP(21)
# train the model
train_model(train_dl, model)
# evaluate the model
acc = evaluate_model(test_dl, model)
print('Accuracy: %.3f' % acc)
