# Ben Lehrburger
# Philanthropy in higher education classifier

# Train and test the neural network

# ***DEPENDENCIES***
import torch
import xlrd
from torch.utils.data import DataLoader
from DataHandling import DonorDataset
from NeuralNetwork import NeuralNetwork
import torch.nn as nn
import torch.optim as optim


# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––– #


# ***NETWORK PARAMETERS***

# Set batch size for training and testing
batch_size = 4
# Set the number of epochs for training
epochs = 50
# Set the learning rate
learning_rate = 0.001
# Set the momentum
momentum = 0.9


# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––– #


# Open the training and testing dataset
loc = '../../Data/base-rate-normalized.xls'
wb = xlrd.open_workbook(loc)

# Training data
train_data_file = wb.sheet_by_index(0)
# Training labels
train_labels_file = wb.sheet_by_index(1)
# Testing data
test_data_file = wb.sheet_by_index(2)
# Testing labels
test_labels_file = wb.sheet_by_index(3)

# Turn training data/labels into actionable data structure
trainers = DonorDataset(train_data_file, train_labels_file)
trainers.parse_file()

# Turn testing data/labels into actionable data structure
testers = DonorDataset(test_data_file, test_labels_file)
testers.parse_file()

# Make data iterable for training and testing
dataloader = DataLoader(trainers, batch_size=batch_size, shuffle=False, num_workers=0)
testloader = DataLoader(testers, batch_size=batch_size, shuffle=False, num_workers=0)

# Create a neural network object
net = NeuralNetwork()


# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––– #


# ***TRAINING***

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Set the learning rate and momentum
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

# Execute each training epoch
for epoch in range(epochs):

    # Store the loss after each training iteration
    running_loss = 0.0

    # Retrieve each data point and associated label
    for i, data in enumerate(dataloader, 0):
        inputs = data['donor'].to(torch.float32)
        labels = data['label'].to(torch.float32)

        optimizer.zero_grad()
        outputs = net(inputs.float())
        labels = labels.type(torch.LongTensor)

        # Compute the error
        loss = criterion(outputs, labels)

        # Back propagate to train weights according to loss
        loss.backward()
        optimizer.step()

        # Print the loss at the current epoch
        running_loss += loss.item()
        if i % 50 == 49:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 50))
            running_loss = 0.0

print('\nFinished Training\n')

# Save the present network's parameters
PATH = 'donation.pth'
torch.save(net.state_dict(), PATH)


# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––– #


# ***TESTING***

total = 0
correct_standard_accuracy = 0
correct_nonlinear_breakdown = 0

# 12 class breakdown
standard_classes = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L')
# 5 class breakdown
constrained_classes = ('$0', '$1-$999', '$1K-$4.9K', '$5K-$24.9K', '$25K+')

# Record correct predictions for each class
correct_standard_classes = {classname: 0 for classname in standard_classes}
total_standard_pred = {classname: 0 for classname in standard_classes}
correct_constrained_classes = {classname: 0 for classname in constrained_classes}
total_constrained_pred = {classname: 0 for classname in constrained_classes}

# Test each testing data point
with torch.no_grad():
    for data in testloader:

        # Save each data point and associated label
        inputs = data['donor'].to(torch.float32)
        labels = data['label'].to(torch.float32)

        # Pass each input through the trained network
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)

        # Loop through each testing label
        for i in range(len(labels)):

            # Check if the network correctly predicted the label in the 12-class breakdown
            if net.is_correct(labels[i], predicted[i]):
                correct_standard_accuracy += 1
                correct_standard_classes[standard_classes[int(predicted[i])]] += 1
            total_standard_pred[standard_classes[int(predicted[i])]] += 1

            # Check if the network correctly predicted the label in the 5-class breakdown
            if net.nonlinear_bracket(labels[i], predicted[i])[0]:
                correct_nonlinear_breakdown += 1
                correct_constrained_classes[constrained_classes[net.nonlinear_bracket(labels[i], predicted[i])[1]]] += 1
            total_constrained_pred[constrained_classes[net.nonlinear_bracket(labels[i], predicted[i])[1]]] += 1

print('Accuracy with standard class breakup: %d %%' % (100 * correct_standard_accuracy / total))

# Calculate 12-class accuracy
for classname, correct_count in correct_standard_classes.items():
    accuracy = 100 * float(correct_count) / total_standard_pred[classname]
    print("Accuracy for class {:2s} is: {:.1f} %".format(classname, accuracy))

print('\nAccuracy with custom nonlinear activation: %d %%' % (100 * correct_nonlinear_breakdown / total))

# Calculate 5-class accuracy
for classname, correct_count in correct_constrained_classes.items():
    accuracy = 100 * float(correct_count) / total_constrained_pred[classname]
    print("Accuracy for class {:10s} is: {:.1f} %".format(classname, accuracy))
