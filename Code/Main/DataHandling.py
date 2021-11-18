# Ben Lehrburger
# Philanthropy in higher education classifier

# Build a dataset object for classifier training and testing

# ***DEPENDENCIES***
import torch
from torch.utils.data import Dataset
import numpy as np

# Wrap a dataset object
class DonorDataset(Dataset):

    def __init__(self, data_file, label_file, transform=None, target_transform=None):

        # Train/test data file
        self.data_file = data_file
        # Train/test labels file
        self.labels_file = label_file

        # Store data
        self.data = []
        # Store labels
        self.labels = []

        # Transform data
        self.transform = transform
        # Transform labels
        self.target_transform = target_transform

    # Parse the given dataset
    def parse_file(self):

        # Extract the relevant features from each data point
        for row in range(0, self.data_file.nrows):

            features = []
            for col in range(0, self.data_file.ncols):
                features.append(float(self.data_file.cell_value(row, col)))
            self.data.append(features)

        self.data = torch.from_numpy(np.array(self.data))

        # Extract the corresponding label for each data point
        for row in range(0, self.labels_file.nrows):

            features = self.labels_file.cell_value(row, 0)
            self.labels.append(float(features))

        self.labels = torch.from_numpy(np.array(self.labels))

    # Retrieve number of data points
    def __len__(self):
        return len(self.labels)

    # Retrieve a specific data point
    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        donor = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            donor = self.transform(donor)
        if self.target_transform:
            label = self.target_transform(label)

        sample = {'donor': donor, 'label': label}

        return sample
