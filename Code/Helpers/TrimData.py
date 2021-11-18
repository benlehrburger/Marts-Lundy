# Ben Lehrburger
# Philanthropy in higher education classifier

# Trim and split datasets

# ***DEPENDENCIES***
import xlrd
from xlwt import Workbook

# Location of dataset file
loc = "../../../Classifier/Data/full-dataset.xls"

wb = xlrd.open_workbook(loc)

# Retrieve each sheet
train_data = wb.sheet_by_index(0)
train_labels = wb.sheet_by_index(1)
test_data = wb.sheet_by_index(2)
test_labels = wb.sheet_by_index(3)

wb = Workbook()

# Write outputs to new file/sheet
trd = wb.add_sheet('Train Data')
trl = wb.add_sheet('Train Labels')
ted = wb.add_sheet('Test Data')
tel = wb.add_sheet('Test Labels')

# Loop over training data
train_row = 0
for row in range(1, train_data.nrows):

    if row % 1000 == 999:
        print('training row ' + str(row))

    if train_data.cell_value(row, 12) != 0:
        id = train_data.cell_value(row, 0)
        trl.write(train_row, 0, id)
        trl.write(train_row, 1, train_labels.cell_value(row, 1))

        for col in range(0, train_data.ncols):
            trd.write(train_row, col, train_data.cell_value(row, col))
        train_row += 1

# Loop over testing data
test_row = 0
for row in range(1, test_data.nrows):

    if row % 1000 == 999:
        print('testing row ' + str(row))

    if test_data.cell_value(row, 12) != 0:
        id = test_data.cell_value(row, 0)
        tel.write(test_row, 0, id)
        tel.write(test_row, 1, test_labels.cell_value(row, 1))

        for col in range(0, test_data.ncols):
            ted.write(test_row, col, test_data.cell_value(row, col))
        test_row += 1

wb.save('../../../Classifier/Data/full-dataset.xls')
