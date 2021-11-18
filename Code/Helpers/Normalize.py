# Ben Lehrburger
# Philanthropy in higher education classifier

# Normalize the data in a dataset

# ***DEPENDENCIES***
import xlrd
from xlwt import Workbook
from sklearn import preprocessing

# Open the dataset
loc = "../../../Classifier/Data/full-dataset.xls"

wb = xlrd.open_workbook(loc)

# Retrieve all data
data = wb.sheet_by_index(4)
# Retrieve all labels
labels = wb.sheet_by_index(5)

wb = Workbook()

# Add sheet with normalized data
normal_data = wb.add_sheet('Data')
normal_labels = wb.add_sheet('Labels')

for col in range(1, data.ncols):

    vals = []
    for row in range(0, data.nrows):
        vals.append(data.cell_value(row, col))

    # Normalize the data
    print("Normalizing...")
    normalized = preprocessing.normalize([vals])

    print("Writing to new file")
    row_num = 0
    for arr in normalized:
        for val in arr:
            normal_data.write(row_num, col, val)
            row_num += 1

print('Done!')

wb.save('../../../Classifier/Data/full-dataset.xls')