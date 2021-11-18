# Ben Lehrburger
# Philanthropy in higher education classifier

# The main executable function
# Prompts user for inputs and predicts the donor's donation bracket
# Inputs given via terminal

# ***DEPENDENCIES***
from NeuralNetwork import NeuralNetwork
import time
import numpy as np
import torch

print("\nThe following are prompts for network inputs. There are 21 required parameters. Please input all values as raw numbers.\n")

time.sleep(2.5)
data = []

num_gifts_last_5_years = input("How many times did this candidate donate over the last 5 years? ")
data.append(int(num_gifts_last_5_years))

num_gifts_last_6_to_10 = input("How many times did this candidate donate between 6 and 10 years ago? ")
data.append(int(num_gifts_last_6_to_10))

dollar_amount_gifts_last_10 = input("What is the dollar amount of this candidate's donations over the last 10 years? ")
data.append(int(dollar_amount_gifts_last_10))

lifetime_hard_commitment = input("What is the dollar amount of this candidate's lifetime hard commitment? ")
data.append(int(lifetime_hard_commitment))

lifetime_soft_commitment = input("What is the dollar amount of this candidate's lifetime soft commitment? ")
data.append(int(lifetime_soft_commitment))

lifetime_commitment = input("What is the dollar amount of this candidate's lifetime commitment? ")
data.append(int(lifetime_commitment))

committees = input("How many school committees has this candidate served on in the last 10 years? ")
data.append(int(committees))

reunions = input("How many school reunions has this candidate attended since graduating? ")
data.append(int(reunions))

sports = input("How many sports did this candidate play at school? ")
data.append(int(sports))

student_activities = input("How many student activities did this candidate participate in at school? ")
data.append(int(student_activities))

degree = input("Did this candidate graduate with a degree? (1 = Yes, 0 = No) ")
data.append(int(degree))

age = input("What is this candidate's age? ")
data.append(int(age))

class_year = input("What is the candidate's class year? ")
class_year = 2018 - int(class_year)
data.append(int(class_year))

c_level_job = input("Does this candidate currently hold a C-level position? (1 = Yes, 0 = No) ")
data.append(int(c_level_job))

primary_affiliation = input("Is this candidate's primary affiliation to the school being an alumni? (1 = Yes, 0 = No) ")
data.append(int(primary_affiliation))

next_reunion = input("Does this candidate have a reunion upcoming in the next 5 years? (1 = Yes, 0 = No) ")
data.append(int(next_reunion))

rfm_score = input("What is this candidate's RFM score? ")
data.append(int(rfm_score))

honors = input("How many honors did this candidate graduate with? ")
data.append(int(honors))

hard_credit = input("What is the dollar amount of the hard credit that this candidate has committed? ")
data.append(int(hard_credit))

soft_credit = input("What is the dollar amount of the soft credit that this candidate has committed? ")
data.append(int(soft_credit))

total_credit = input("What is the dollar amount of the total credit that this candidate has committed? ")
data.append(int(total_credit))

time.sleep(1)
print("\nThe classifier will now predict this candidate's projected donations over the next 5 years\n")
time.sleep(1)

PATH = 'donation.pth'

# Classifier classes
classes = ('$0', '$1-$999', '$1K-$4.9K', '$5K-$24.9K', '$25K+')

# Confidence in each class prediction
confidence = ('72.5%', '78.70%', '74.10%', '67.20%', '83.30%')

model = NeuralNetwork()
model.load_state_dict(torch.load(PATH))
model.eval()

# Means of each feature
means = [3.461397059, 2.441176471, 109854.8343, 143049.0391, 7046.33955, 150095.3786, 0.433517157, 4.620404412,
         0.340073529, 1.668198529, 0.686887255, 56.06403186, 33.06403186, 0.407781863, 0.709865196, 0.649509804,
         1.26807598, 0.28125, 18739.04175, 90.06041973, 18829.10217]

# Standard deviations of each feature
standard_deviations = [1.679785171, 2.120540489, 890334.4198, 1271191.472, 213164.2716, 1295540.188, 1.106575397,
                       4.880699983, 0.642388276, 1.320488376, 0.46383086, 17.30021565, 17.30021565, 0.491497534,
                       0.45389395, 0.477196589, 0.523558839, 0.578449559, 135597.9022, 790.8485327, 135599.9779]


# Normalize the inputted data
def normalize(sample):

    normalized_sample = []

    for datapoint in sample:
        index = sample.index(datapoint)
        normalized_value = (float(datapoint) - means[index]) / standard_deviations[index]
        normalized_sample.append(normalized_value)

    return np.array(normalized_sample)


# Propagate the user's input through the network and get an output
def get_prediction(sample):

    # Normalize the data
    normalized_data = normalize(sample)
    tensor = torch.from_numpy(normalized_data).double()

    # Propagate user input forwards
    output = model.forward(tensor.to(torch.float32))

    # Max of outputs is predicted class
    y_hat = output.max(0)

    return y_hat


index = get_prediction(data).indices.numpy()
bracket = model.classify(index)

print('We predict that this candidate will give ' + str(classes[bracket]) + ' over the next 5 years with ' +
      str(confidence[bracket] + ' confidence'))
