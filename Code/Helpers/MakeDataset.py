# Ben Lehrburger
# Philanthropy in higher education classifier

# Make a dataset in Excel
# Scrubs sensitive information and encodes data

# ***DEPENDENCIES***
import xlrd
from xlwt import Workbook
from MakeDonor import Donor
from Shortcuts import get_ids, categorize, clean

# ***INPUTS***
training_samples = 40000
testing_samples = 4000


# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––– #


# Location of un-parsed dataset
loc = "../../../Classifier/Data/unparsed-table.xls"

wb = xlrd.open_workbook(loc)

# Open each sheet in the dataset
entity = wb.sheet_by_index(0)
honors = wb.sheet_by_index(1)
gifts = wb.sheet_by_index(2)

# Loop over each column in each dataset
entity_col = range(entity.ncols)
honors_col = range(honors.ncols)
gifts_col = range(gifts.ncols)

total_samples = training_samples + testing_samples

# Loop over each row in each dataset
entity_row = range(0, total_samples+1)
honors_row = range(0, total_samples+1)
gifts_row = range(0, total_samples+1)


# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––– #


# Grab the donor IDs from each table
honors_ids = get_ids(honors)
gift_ids = get_ids(gifts)

# Hold the total categories from each table
categories = []

# Get categories from each table
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
        print('Donor ' + str(row))

print('Encoding...')

# Grab each data point
entity_ids = []
for r in entity_row:
    if r != 0:
        entity_ids.append(entity.cell_value(r, 0))

# Scrub and encode each data point
donors = clean(donors, entity_ids)

print('Writing to new files...')

# Write donor dictionary to new Excel file
wb = Workbook()

# Training data
train_data = wb.add_sheet('Train Data')
# Training labels
train_labels = wb.add_sheet('Train Labels')
# Test data
test_data = wb.add_sheet('Test Data')
# Test labels
test_labels = wb.add_sheet('Test Labels')

# Write category headers in their respective files
train_data.write(0, 0, 'ID')
train_labels.write(0, 0, 'ID')
test_data.write(0, 0, 'ID')
test_labels.write(0, 0, 'ID')

for category in categories[:-1]:
    train_data.write(0, categories.index(category)+1, category)
    test_data.write(0, categories.index(category)+1, category)

train_labels.write(0, 1, 'Total Commitment Bin')
test_labels.write(0, 1, 'Total Commitment Bin')

# Write each data point to new file
# Separate files according to user-defined number of training and testing samples
for id in entity_ids:
    if entity_ids.index(id) < training_samples:

        train_data.write(entity_ids.index(id)+1, 0, id)
        train_labels.write(entity_ids.index(id)+1, 0, id)

        for category in categories[:-1]:
            train_data.write(entity_ids.index(id)+1, categories.index(category)+1, donors[entity_ids.index(id)][str(id)]
            [category])
        train_labels.write(entity_ids.index(id)+1, 1, donors[entity_ids.index(id)][str(id)]['Total Commitment Bin'])

    else:
        test_data.write(entity_ids.index(id)+1-training_samples, 0, id)
        test_labels.write(entity_ids.index(id)+1-training_samples, 0, id)

        for category in categories[:-1]:
            test_data.write(entity_ids.index(id)+1-training_samples, categories.index(category)+1,
                            donors[entity_ids.index(id)][str(id)][category])
        test_labels.write(entity_ids.index(id)+1-training_samples, 1, donors[entity_ids.index(id)][str(id)]
        ['Total Commitment Bin'])

# Save the new dataset
wb.save('../../../Classifier/Data/dataset.xls')

print('Great success!')
