# Get the ids from each table
def get_ids(table):
    ids = {}
    row_range = range(0, table.nrows)
    for r in row_range:
        print(r)
        if r != 0:
            if int(table.cell_value(r, 0)) in ids:
                ids[int(table.cell_value(r, 0))].append(r)
            else:
                ids[int(table.cell_value(r, 0))] = [r]
    return ids


# Add categories from each table to dictionaries
def categorize(table, categories):
    cats = []
    for col in range(1, table.ncols):
        if table.cell_value(0, col) not in categories:
            cats.append(table.cell_value(0, col))
            categories.append(table.cell_value(0, col))
    return cats
