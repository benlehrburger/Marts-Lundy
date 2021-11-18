# Ben Lehrburger
# Philanthropy in higher education classifier

# Helper functions used across the classifier

# Get the ids from each table
def get_ids(table):

    ids = {}
    row_range = range(0, table.nrows)

    for r in row_range:

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


# Encode all entries in donor dictionary as floating point values
def clean(donors, entity_ids):

    for id in entity_ids:
        str_id = str(id)
        print('scrubbing donor ' + str_id)
        index = entity_ids.index(id)

        for category in donors[index][str_id]:

            # ***STANDARD CONVERSIONS***

            if donors[index][str_id][category] == []:
                donors[index][str_id][category] = 0.0

            if donors[index][str_id][category] == 'No':
                donors[index][str_id][category] = 0.0

            if donors[index][str_id][category] == 'No Employment Data':
                donors[index][str_id][category] = 0.0

            if donors[index][str_id][category] == '':
                donors[index][str_id][category] = 0.0

            if donors[index][str_id][category] == 'None':
                donors[index][str_id][category] = 0.0

            if donors[index][str_id][category] == 'NA':
                donors[index][str_id][category] = 0.0

            if donors[index][str_id][category] == 'Yes':
                donors[index][str_id][category] = 1.0

            # ***SPECIFIC CONVERSIONS***

            # 1 if alumni, otherwise 0
            if category == 'Primary Affiliation':
                if donors[index][str_id][category] == 'Alumnus/a':
                    donors[index][str_id][category] = 1.0
                else:
                    donors[index][str_id][category] = 0.0

            # Number reflects amount of honor roles
            if (category == 'Honor') and (donors[index][str_id][category] != 0.0) and (isinstance(donors[index][str_id]
                                                                                                  [category], list)):
                donors[index][str_id][category] = float(len(donors[index][str_id][category]))

            # Year of dataset creation (2018) minus class year
            if (category == 'Class Year') and (donors[index][str_id][category] != 0.0):
                donors[index][str_id][category] = 2018 - donors[index][str_id][category]

            # 1 if reunion in next 5 years, otherwise 0
            if (category == 'Next Reunion') and (donors[index][str_id][category] != 0.0):
                donors[index][str_id][category] = 1.0

            # Extract RFM score –– 1 = high, 2 = middle, 3 = low
            if category == 'RFM Score':
                while str(donors[index][str_id][category])[:1] == ' ':
                    donors[index][str_id][category] = str(donors[index][str_id][category])[1:]
                donors[index][str_id][category] = float(str(donors[index][str_id][category])[:1])

            # Get sum of gifts category
            if (category == 'Total Commitment') and (donors[index][str_id][category] != 0.0) and \
                    (isinstance(donors[index][str_id][category], list)):

                if len(donors[index][str_id][category]) > 1:
                    total = 0.0

                    for i in donors[index][str_id][category]:
                        total += i
                    donors[index][str_id][category] = total

                else:
                    donors[index][str_id][category] = donors[index][str_id][category][0]

            # Get sum of hard credit category
            if (category == 'Hard Credit') and (donors[index][str_id][category] != 0.0) and \
                    (isinstance(donors[index][str_id][category], list)):

                if len(donors[index][str_id][category]) > 1:
                    total = 0.0

                    for i in donors[index][str_id][category]:
                        total += i
                    donors[index][str_id][category] = total

                else:
                    donors[index][str_id][category] = donors[index][str_id][category][0]

            # Get sum of soft credit category
            if (category == 'Soft Credit') and (donors[index][str_id][category] != 0.0):

                if len(donors[index][str_id][category]) > 1:
                    total = 0.0

                    for i in donors[index][str_id][category]:
                        total += i
                    donors[index][str_id][category] = total

                else:
                    donors[index][str_id][category] = donors[index][str_id][category][0]

            # Encode total commitment (y_hat) by bin for classifier
            if (category == 'Total Commitment Bin') and (donors[index][str_id][category] != 0.0) and not \
                    (isinstance(donors[index][str_id][category], float)):

                for i in donors[index][str_id][category]:
                    donors[index][str_id][category][donors[index][str_id][category].index(i)] = i[:1]
                donors[index][str_id][category] = min(donors[index][str_id][category])

                # A = 25K+
                if donors[index][str_id][category] == 'A':
                    donors[index][str_id][category] = 11.0

                # B = 10K-24K
                elif donors[index][str_id][category] == 'B':
                    donors[index][str_id][category] = 10.0

                # C = 5K-9K
                elif donors[index][str_id][category] == 'C':
                    donors[index][str_id][category] = 9.0

                # D = 2.5K-4.9K
                elif donors[index][str_id][category] == 'D':
                    donors[index][str_id][category] = 8.0

                # E = 1K-2.4K
                elif donors[index][str_id][category] == 'E':
                    donors[index][str_id][category] = 7.0

                # F = 500-999
                elif donors[index][str_id][category] == 'F':
                    donors[index][str_id][category] = 6.0

                # G = 250-499
                elif donors[index][str_id][category] == 'G':
                    donors[index][str_id][category] = 5.0

                # H = 100-249
                elif donors[index][str_id][category] == 'H':
                    donors[index][str_id][category] = 4.0

                # I = 50-99
                elif donors[index][str_id][category] == 'I':
                    donors[index][str_id][category] = 3.0

                # J = 25-49
                elif donors[index][str_id][category] == 'J':
                    donors[index][str_id][category] = 2.0

                # K = 1-24
                elif donors[index][str_id][category] == 'K':
                    donors[index][str_id][category] = 1.0

                # L = 0
                else:
                    donors[index][str_id][category] = 0.0

    return donors
