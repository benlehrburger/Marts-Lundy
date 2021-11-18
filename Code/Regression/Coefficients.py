# Ben Lehrburger
# Philanthropy in higher education classifier

# Calculate the weights for use in regression analysis

# ***DEPENDENCIES***
import xlrd
import numpy as np
import time
from Classifier.Code.Helpers.MakeDonor import Donor
from Classifier.Code.Helpers.Shortcuts import *

# Location of dataset
loc = "../../../Classifier/Data/unparsed-table.xls"

wb = xlrd.open_workbook(loc)

# Open each table
entity = wb.sheet_by_index(0)
honors = wb.sheet_by_index(1)
gifts = wb.sheet_by_index(2)

# Loop over each column in each dataset
entity_col = range(entity.ncols)
honors_col = range(honors.ncols)
gifts_col = range(gifts.ncols)

# Loop over each row in each dataset
entity_row = range(entity.nrows)
honors_row = range(honors.nrows)
gifts_row = range(gifts.nrows)


# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––– #


# Get IDs from each table
honors_ids = get_ids(honors)
gift_ids = get_ids(gifts)

# Get categories from each table
categories = []
entity_cats = categorize(entity, categories)
honors_cats = categorize(honors, categories)
gift_cats = categorize(gifts, categories)

# Initialize an array to hold all our donors
donors = []

# Create each donor in the entity table
for row in entity_row:
    
    if row != 0:
        donors.append(Donor(row, entity, honors, gifts, honors_ids, gift_ids, entity_cats, honors_cats, gift_cats,
                            categories).makeDonor())
        print('Creating donor ' + str(row))

print('Encoding...')

# Scrub and encode each data point
entity_ids = []
for r in entity_row:
    
    if r != 0:
        entity_ids.append(entity.cell_value(r, 0))

donors = clean(donors, entity_ids)

print('Done building dictionary! Starting to transfer into matrices...')

# Initialize matrix A
A = np.zeros((len(entity_ids), len(categories)))

# Add each donor's features to matrix A
for id in entity_ids:
    
    print('Adding donor ' + str(id) + ' to A matrix')
    
    for category in categories[:-1]:
        A[entity_ids.index(id)][categories.index(category)] = donors[entity_ids.index(id)][str(id)][category]
    A[entity_ids.index(id)][-1] = 1.0

noise = np.random.rand(A.shape[0], A.shape[1]) / 100
A += noise

# Initialize matrix b
b = -1 * np.ones((len(entity_ids), 12))
for id in entity_ids:
    
    print('Adding donor ' + str(id) + ' to b matrix')
    
    # Add each label to matrix b
    bin = int(float(donors[entity_ids.index(id)][str(id)]['Total Commitment Bin']))
    b[entity_ids.index(id)][bin] = 1

# Solve normal equations
print('Solving normal equations...')
coefficients = np.linalg.solve(A.T @ A, A.T @ b)
print(coefficients)

print('Great success!')
