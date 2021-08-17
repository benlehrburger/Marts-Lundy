import math
import numpy as np
import xlrd

#
# Implement linear layer forward and backward prop
#

class LinearLayer:
    def __init__(self, _m, _n):
        '''
         * :param _m: _m is the input X hidden size
         * :param _n: _n is the output Y hidden size
        '''
        # "Kaiming initialization" is important for neural network to converge. The NN will not converge without it!
        self.W = (np.random.uniform(low=-10000.0, high=10000.0, size=(_m, _n))) / 10000.0 * np.sqrt(6.0 / _m)
        self.stored_X = None
        self.W_grad = None  # record the gradient of the weight

    def forward(self, X):
        '''
         * :param X: shape(X)[0] is batch size and shape(X)[1] is the #features
         * Store the input X in stored_data for Backward.
         * :return: X * weights
        '''

        # store input X in stored_data for backward
        X = X.astype(np.float)
        self.stored_X = X
        # return X*weights
        return X @ self.W

    def backward(self, Y_grad):
        '''
         * shape(output_grad)[0] is batch size and shape(output_grad)[1] is the # output features (shape(weight)[1])
         * Calculate the gradient of the output (the result of the Forward method) w.r.t. the **W** and store the product of the gradient and Y_grad in W_grad
         * Calculate the gradient of the output (the result of the Forward method) w.r.t. the **X** and return the product of the gradient and Y_grad
        '''

        # calculate gradient of output with respect to weights, multiply by the gradient of Y, and store
        self.W_grad = self.stored_X.T @ Y_grad
        # calculate gradient of output with respect to X, multiply by the gradient of Y, and return
        return Y_grad @ self.W.T

class ReLU:
    # sigmoid layer
    def __init__(self):
        self.stored_X = None  # Here we should store the input matrix X for Backward

    def forward(self, X):
        '''
         *  The input X matrix has the dimension [#samples, #features].
         *  The output Y matrix has the same dimension as the input X.
         *  Create an output matrix by going through each element in input and calculate relu=max(0,x) and
         *  Store the input X in self.stored_X for Backward.
        '''

        # store input X in stored_data for Backward
        self.stored_X = X

        # initialize output matrix Y with same dimensions as input X
        Y = np.zeros((X.shape[0], X.shape[1]))
        # initialize row counter
        row_counter = 0
        # for each row in input matrix X
        for row in X:
            # initialize column counter
            column_counter = 0
            # for each element in each row
            for element in row:
                # the value of that element's position in ouput matrix Y is its ReLU activation function
                Y[row_counter][column_counter] = max(0, element)
                column_counter += 1
            row_counter += 1

        # return ReLU output matrix Y
        return Y

    def backward(self, Y_grad):
        '''
         *  grad_relu(x)=1 if relu(x)=x
         *  grad_relu(x)=0 if relu(x)=0
         *
         *  The input matrix has the name "output_grad." The name is confusing (it is actually the input of the function)
         *  The output matrix has the same dimension as input.
         *  The output matrix is calculated as grad_relu(stored_X)*Y_grad.
         *  returns the output matrix calculated above
        '''

        # initialize output matrix with same dimensions as matrix holding gradient of Y
        output = np.zeros((Y_grad.shape[0], Y_grad.shape[1]))
        # initialize row counter
        row_counter = 0
        # for each row in input matrix holding gradients of Y
        for row in Y_grad:
            # initialize column counter
            column_counter = 0
            # for each element in each row
            for element in row:
                # if that element's original value is greater than 0
                if self.stored_X[row_counter][column_counter] > 0:
                    # set its value in the output matrix to its gradient with respect to Y
                    output[row_counter][column_counter] = Y_grad[row_counter][column_counter]
                # otherwise leave it as 0
                else:
                    None
                column_counter += 1
            row_counter += 1

        # return the gradient with respect to X
        return output

#
# Implement MSE loss function
#

class MSELoss:
    # cross entropy loss
    # return the mse loss mean(y_j-y_pred_i)^2

    def __init__(self):
        self.stored_diff = None

    def forward(self, prediction, groundtruth):
        '''
         *  Calculate stored_data=pred-truth
         *  Calculate the MSE loss as the squared sum of all the elements in the stored_data divided by the number of elements, i.e., MSE(pred, truth) = ||pred-truth||^2 / N, with N as the total number of elements in the matrix
        '''

        # store difference as that between prediction and groundtruth
        self.stored_diff = prediction - groundtruth

        # initialize loss and row counter as 0
        mse_loss = 0
        # for each row in the difference matrix
        for row in range(self.stored_diff.shape[0]):
            # for each element in each row
            for column in range(self.stored_diff.shape[1]):
                # add to the existing loss the squared sum of the prediction minus groundtruth
                mse_loss += (self.stored_diff[row][column]) ** 2

                # return the squared sum loss divided by the number of samples
        return mse_loss / self.stored_diff.size

    # return the gradient of the input data
    def backward(self):
        '''
         * return the gradient matrix of the MSE loss
         * The output matrix has the same dimension as the stored_data (make sure you have stored the (pred-truth) in stored_data in your forward function!)
         * Each element (i,j) of the output matrix is calculated as grad(i,j)=2(pred(i,j)-truth(i,j))/N
        '''

        # initialize matrix of zeros with the same dimensions as the difference matrix
        mse_grad = np.zeros((self.stored_diff.shape[0], self.stored_diff.shape[1]))

        # for each row in the output matrix
        for row in range(self.stored_diff.shape[0]):
            # for each element in each row
            for column in range(self.stored_diff.shape[1]):
                # set that element's position in the gradient matrix
                mse_grad[row][column] = self.stored_diff[row][column]

        # return the output matrix multiplied by the gradient of the loss function with respect to our prediction
        return mse_grad * (2 / self.stored_diff.size)

class Network:
    def __init__(self, layers_arch):
        '''
         *  Initialize the array for input layers with the proper feature sizes specified in the input vector.
         *  For the linear layer, in each pair (in_size, out_size), the in_size is the feature size of the previous layer and the out_size is the feature size of the output (that goes to the next layer)
         *  In the linear layer, the weight should have the shape (in_size, out_size).
         * The output feature size of the linear layer i should always equal to the input feature size of the linear layer i+1.
        '''

        # initialize empty array to hold layers
        self.layers = []

        # for each layer in the given structure
        for layer in layers_arch:
            # if that layer is linear
            if layer[0] == 'Linear':
                # add a linear layer to our array with input size equal to that specified in the given structure
                self.layers.append(LinearLayer(layer[1][0], layer[1][1]))
            # otherwise add a ReLU layer
            else:
                self.layers.append(ReLU())

    def forward(self, X):
        '''
         * propagate the input data for the first linear layer throught all the layers in the network and return the output of the last linear layer.
         * For implementation, you need to write a for-loop to propagate the input from the first layer to the last layer (before the loss function) by going through the forward functions of all the layers.
         * For example, for a network with k linear layers and k-1 activation layers, the data flow is:
         * linear[0] -> activation[0] -> linear[1] ->activation[1] -> ... -> linear[k-2] -> activation[k-2] -> linear[k-1]
        '''

        # current input is equal to given dataset
        current = X
        # for each layer in the array of layers
        for layer in self.layers:
            # current input is equal to the output of the previous layer
            current = layer.forward(current)

        # return the final output
        return current

    def backward(self, Y_grad):
        '''
         * Propagate the gradient from the last layer to the first layer by going through the backward functions of all the layers.
         * propagate the gradient of the output (we got from the Forward method) back throught the network and return the gradient of the first layer.
         * Notice: We should use the chain rule for the backward.
         * Notice: The order is opposite to the forward.
        '''

        # current input is equal to the gradient of our prediction
        current = Y_grad
        # for each layer in the reversed network
        for layer in list(reversed(self.layers)):
            # take the gradient of each layer with respect to the previous layer
            current = layer.backward(current)

        # return the final output
        return current

def One_Hot_Encode(labels, classes=12):
    '''
     *  Make the labels one-hot.
     *  For example, if there are 5 classes {0, 1, 2, 3, 4} then
     *  [0, 2, 4] -> [[1, 0, 0, 0, 0],
     * 				  [0, 0, 1, 0, 0],
     * 				  [0, 0, 0, 0, 1]]
    '''

    # initialize matrix of zeros with dimensions labels x classes
    one_hot = np.zeros((len(labels), classes))
    # initialize label counter to track rows
    label_counter = 0
    # for each value in the array of labels
    for val in labels:
        # make that label's value in a unique row equal to 1
        one_hot[label_counter][int(val)] = 1
        label_counter += 1

    # return one-hot encoded array
    return one_hot

#
# Implement classifier with gradient descent
#

class Classifier:
    # Classifier
    def __init__(self, train_data_file, train_labels_file, test_data_file, test_labels_file, layers_arch,
                 learning_rate=1e-3, batch_size=32, max_epoch=200, classes=12):
        self.classes = classes

        self.train_data_file = train_data_file
        self.train_labels_file = train_labels_file
        self.test_data_file = test_data_file
        self.test_labels_file = test_labels_file

        self.train_data = []  # The shape of train data should be (n_samples,28^2)
        self.train_labels = []
        self.test_data = []
        self.test_labels = []

        self.layers_arch = layers_arch
        self.net = Network(layers_arch)
        self.loss_function = MSELoss()

        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def dataloader(self):

        # Training data
        for row in range(1, self.train_data_file.nrows):
            features = []
            for col in range(1, self.train_data_file.ncols):
                features.append(self.train_data_file.cell_value(row, col))
            self.train_data.append(features)
        self.train_data = np.array(self.train_data)

        # Training labels
        for row in range(1, self.train_labels_file.nrows):
            features = self.train_labels_file.cell_value(row, 1)
            self.train_labels.append(features)
        self.train_labels = np.array(self.train_labels)

        # Testing data
        for row in range(1, self.test_data_file.nrows):
            features = []
            for col in range(1, self.test_data_file.ncols):
                features.append(self.test_data_file.cell_value(row, col))
            self.test_data.append(features)
        self.test_data = np.array(self.test_data)

        # Testing labels
        for row in range(1, self.test_labels_file.nrows):
            features = [self.test_labels_file.cell_value(row, 1)]
            self.test_labels.append(features)
        self.test_labels = np.array(self.test_labels)

    def Train_One_Epoch(self):
        '''
         * Here we train the network using gradient descent
        '''
        loss = 0
        n_loop = int(math.ceil(len(self.train_data) / self.batch_size))
        for i in range(n_loop):
            batch_data = self.train_data[i * self.batch_size: (i + 1) * self.batch_size]
            batch_label = self.train_labels[i * self.batch_size: (i + 1) * self.batch_size]
            batch_one_hot_label = One_Hot_Encode(batch_label, classes=self.classes)

            '''
             *  Forward the data to the network.
             *  Forward the result to the loss function.
             *  Backward.
             *  Update the weights with weight gradients.
             *  Do not forget the learning rate!
            '''

            # propagate input data forwards and store the prediction
            prediction = self.net.forward(batch_data)
            # calculate the squared sum of all prediction and groundtruth differences
            loss += self.loss_function.forward(prediction, batch_one_hot_label)
            # propogate the gradient of the prediction backwards
            pred_grad = self.loss_function.backward()
            self.net.backward(pred_grad)
            # for each layer in the network
            for layer in range(len(self.layers_arch)):
                # if that layer is linear
                if self.layers_arch[layer][0] == 'Linear':
                    # multiply negative gradients by the learning rate
                    self.net.layers[layer].W -= self.net.layers[layer].W_grad * self.learning_rate

        return loss / n_loop

    def Test(self):
        '''
         * the class with max score is our predicted label
        '''
        score = self.net.forward(self.test_data)
        accuracy = 0

        for i in range(np.shape(score)[0]):
            one_label_list = score[i].tolist()
            label_pred = one_label_list.index(max(one_label_list))
            if label_pred == self.test_labels[i]:
                accuracy = accuracy + 1

        accuracy = accuracy / np.shape(score)[0]
        return accuracy

    def Train(self):
        self.dataloader()
        for i in range(self.max_epoch):
            loss = self.Train_One_Epoch()
            accuracy = self.Test()
            print("Epoch: ", str(i + 1), "/", str(self.max_epoch), " | Train loss: ", loss, " | Test Accuracy : ", accuracy)


loc = 'test-dataset.xls'
wb = xlrd.open_workbook(loc)
train_data_file = wb.sheet_by_index(0)
train_labels_file = wb.sheet_by_index(1)
test_data_file = wb.sheet_by_index(2)
test_labels_file = wb.sheet_by_index(3)

classifier_layers_arch = [['Linear', (21, 256)], ['ReLU'], ['Linear', (256, 12)]]
cls = Classifier(train_data_file, train_labels_file, test_data_file, test_labels_file, layers_arch = classifier_layers_arch, learning_rate = 0.01, batch_size = 32, max_epoch = 200)
cls.Train()
cls.Test()

