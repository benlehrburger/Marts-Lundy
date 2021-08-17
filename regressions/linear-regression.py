import numpy as np

donors = [{'17.0': {'#GiftsL5Y': 0.0, '#GiftsL6-10Y': 0.0, '$GivingL10Y': 0.0, 'Total Lifetime Hard Commitment': 765.0, 'Total Lifetime Soft Commitment': 0.0, 'Total Lifetime Commitment': 765.0, '#CommitteesL10Y': 5.0, '#Reunions': 0.0, '#Sports': 0.0, '#Student Activities': 10.0, '?Has Degree?': 1.0, 'Age': 30.0, 'Class Year': 7.0, 'C Level Job?': 0.0, 'Primary Affiliation': 1.0, 'Next Reunion': 1.0, 'RFM Score': '3', 'Honor': 0.0, 'Hard Credit': 0.0, 'Soft Credit': 0.0, 'Total Commitment': 0.0, 'Total Commitment Bin': 0.0}}, {'18.0': {'#GiftsL5Y': 0.0, '#GiftsL6-10Y': 2.0, '$GivingL10Y': 100.0, 'Total Lifetime Hard Commitment': 180.0, 'Total Lifetime Soft Commitment': 0.0, 'Total Lifetime Commitment': 180.0, '#CommitteesL10Y': 1.0, '#Reunions': 0.0, '#Sports': 0.0, '#Student Activities': 3.0, '?Has Degree?': 1.0, 'Age': 30.0, 'Class Year': 7.0, 'C Level Job?': 0.0, 'Primary Affiliation': 1.0, 'Next Reunion': 1.0, 'RFM Score': '2', 'Honor': 0.0, 'Hard Credit': 0.0, 'Soft Credit': 0.0, 'Total Commitment': 0.0, 'Total Commitment Bin': 0.0}}, {'20.0': {'#GiftsL5Y': 0.0, '#GiftsL6-10Y': 0.0, '$GivingL10Y': 0.0, 'Total Lifetime Hard Commitment': 125.0, 'Total Lifetime Soft Commitment': 0.0, 'Total Lifetime Commitment': 125.0, '#CommitteesL10Y': 2.0, '#Reunions': 0.0, '#Sports': 0.0, '#Student Activities': 8.0, '?Has Degree?': 1.0, 'Age': 30.0, 'Class Year': 7.0, 'C Level Job?': 0.0, 'Primary Affiliation': 1.0, 'Next Reunion': 1.0, 'RFM Score': '3', 'Honor': 2.0, 'Hard Credit': 0.0, 'Soft Credit': 0.0, 'Total Commitment': 0.0, 'Total Commitment Bin': 0.0}}, {'23.0': {'#GiftsL5Y': 1.0, '#GiftsL6-10Y': 0.0, '$GivingL10Y': 125.0, 'Total Lifetime Hard Commitment': 125.0, 'Total Lifetime Soft Commitment': 0.0, 'Total Lifetime Commitment': 125.0, '#CommitteesL10Y': 2.0, '#Reunions': 0.0, '#Sports': 0.0, '#Student Activities': 4.0, '?Has Degree?': 1.0, 'Age': 30.0, 'Class Year': 7.0, 'C Level Job?': 0.0, 'Primary Affiliation': 1.0, 'Next Reunion': 1.0, 'RFM Score': '2', 'Honor': 1.0, 'Hard Credit': 125.0, 'Soft Credit': 0.0, 'Total Commitment': 125.0, 'Total Commitment Bin': 4.0}}, {'24.0': {'#GiftsL5Y': 1.0, '#GiftsL6-10Y': 0.0, '$GivingL10Y': 100.0, 'Total Lifetime Hard Commitment': 100.0, 'Total Lifetime Soft Commitment': 0.0, 'Total Lifetime Commitment': 100.0, '#CommitteesL10Y': 0.0, '#Reunions': 0.0, '#Sports': 0.0, '#Student Activities': 1.0, '?Has Degree?': 1.0, 'Age': 30.0, 'Class Year': 7.0, 'C Level Job?': 0.0, 'Primary Affiliation': 1.0, 'Next Reunion': 1.0, 'RFM Score': '2', 'Honor': 1.0, 'Hard Credit': 0.0, 'Soft Credit': 0.0, 'Total Commitment': 0.0, 'Total Commitment Bin': 0.0}}, {'27.0': {'#GiftsL5Y': 2.0, '#GiftsL6-10Y': 0.0, '$GivingL10Y': 125.0, 'Total Lifetime Hard Commitment': 125.0, 'Total Lifetime Soft Commitment': 0.0, 'Total Lifetime Commitment': 125.0, '#CommitteesL10Y': 0.0, '#Reunions': 0.0, '#Sports': 0.0, '#Student Activities': 1.0, '?Has Degree?': 1.0, 'Age': 30.0, 'Class Year': 7.0, 'C Level Job?': 0.0, 'Primary Affiliation': 1.0, 'Next Reunion': 1.0, 'RFM Score': '2', 'Honor': 1.0, 'Hard Credit': 0.0, 'Soft Credit': 0.0, 'Total Commitment': 0.0, 'Total Commitment Bin': 0.0}}, {'30.0': {'#GiftsL5Y': 0.0, '#GiftsL6-10Y': 4.0, '$GivingL10Y': 150.0, 'Total Lifetime Hard Commitment': 200.0, 'Total Lifetime Soft Commitment': 0.0, 'Total Lifetime Commitment': 200.0, '#CommitteesL10Y': 1.0, '#Reunions': 0.0, '#Sports': 1.0, '#Student Activities': 5.0, '?Has Degree?': 1.0, 'Age': 30.0, 'Class Year': 7.0, 'C Level Job?': 0.0, 'Primary Affiliation': 1.0, 'Next Reunion': 1.0, 'RFM Score': '2', 'Honor': 1.0, 'Hard Credit': 0.0, 'Soft Credit': 0.0, 'Total Commitment': 0.0, 'Total Commitment Bin': 0.0}}, {'31.0': {'#GiftsL5Y': 1.0, '#GiftsL6-10Y': 0.0, '$GivingL10Y': 135.0, 'Total Lifetime Hard Commitment': 135.0, 'Total Lifetime Soft Commitment': 0.0, 'Total Lifetime Commitment': 135.0, '#CommitteesL10Y': 0.0, '#Reunions': 0.0, '#Sports': 0.0, '#Student Activities': 2.0, '?Has Degree?': 1.0, 'Age': 30.0, 'Class Year': 7.0, 'C Level Job?': 0.0, 'Primary Affiliation': 1.0, 'Next Reunion': 1.0, 'RFM Score': '2', 'Honor': 0.0, 'Hard Credit': 135.0, 'Soft Credit': 0.0, 'Total Commitment': 135.0, 'Total Commitment Bin': 4.0}}, {'35.0': {'#GiftsL5Y': 0.0, '#GiftsL6-10Y': 0.0, '$GivingL10Y': 0.0, 'Total Lifetime Hard Commitment': 175.0, 'Total Lifetime Soft Commitment': 0.0, 'Total Lifetime Commitment': 175.0, '#CommitteesL10Y': 1.0, '#Reunions': 0.0, '#Sports': 0.0, '#Student Activities': 3.0, '?Has Degree?': 1.0, 'Age': 30.0, 'Class Year': 7.0, 'C Level Job?': 0.0, 'Primary Affiliation': 1.0, 'Next Reunion': 1.0, 'RFM Score': '3', 'Honor': 1.0, 'Hard Credit': 0.0, 'Soft Credit': 0.0, 'Total Commitment': 0.0, 'Total Commitment Bin': 0.0}}, {'37.0': {'#GiftsL5Y': 1.0, '#GiftsL6-10Y': 0.0, '$GivingL10Y': 125.0, 'Total Lifetime Hard Commitment': 125.0, 'Total Lifetime Soft Commitment': 0.0, 'Total Lifetime Commitment': 125.0, '#CommitteesL10Y': 0.0, '#Reunions': 0.0, '#Sports': 1.0, '#Student Activities': 1.0, '?Has Degree?': 1.0, 'Age': 30.0, 'Class Year': 7.0, 'C Level Job?': 0.0, 'Primary Affiliation': 1.0, 'Next Reunion': 1.0, 'RFM Score': '2', 'Honor': 1.0, 'Hard Credit': 125.0, 'Soft Credit': 0.0, 'Total Commitment': 125.0, 'Total Commitment Bin': 4.0}}, {'38.0': {'#GiftsL5Y': 0.0, '#GiftsL6-10Y': 0.0, '$GivingL10Y': 0.0, 'Total Lifetime Hard Commitment': 100.0, 'Total Lifetime Soft Commitment': 0.0, 'Total Lifetime Commitment': 100.0, '#CommitteesL10Y': 0.0, '#Reunions': 0.0, '#Sports': 0.0, '#Student Activities': 2.0, '?Has Degree?': 1.0, 'Age': 30.0, 'Class Year': 7.0, 'C Level Job?': 0.0, 'Primary Affiliation': 1.0, 'Next Reunion': 1.0, 'RFM Score': '3', 'Honor': 0.0, 'Hard Credit': 0.0, 'Soft Credit': 0.0, 'Total Commitment': 0.0, 'Total Commitment Bin': 0.0}}, {'39.0': {'#GiftsL5Y': 0.0, '#GiftsL6-10Y': 0.0, '$GivingL10Y': 0.0, 'Total Lifetime Hard Commitment': 250.0, 'Total Lifetime Soft Commitment': 0.0, 'Total Lifetime Commitment': 250.0, '#CommitteesL10Y': 1.0, '#Reunions': 0.0, '#Sports': 0.0, '#Student Activities': 7.0, '?Has Degree?': 1.0, 'Age': 30.0, 'Class Year': 7.0, 'C Level Job?': 0.0, 'Primary Affiliation': 1.0, 'Next Reunion': 1.0, 'RFM Score': '3', 'Honor': 0.0, 'Hard Credit': 0.0, 'Soft Credit': 0.0, 'Total Commitment': 0.0, 'Total Commitment Bin': 0.0}}, {'42.0': {'#GiftsL5Y': 1.0, '#GiftsL6-10Y': 0.0, '$GivingL10Y': 200.0, 'Total Lifetime Hard Commitment': 200.0, 'Total Lifetime Soft Commitment': 0.0, 'Total Lifetime Commitment': 200.0, '#CommitteesL10Y': 3.0, '#Reunions': 0.0, '#Sports': 0.0, '#Student Activities': 2.0, '?Has Degree?': 1.0, 'Age': 30.0, 'Class Year': 7.0, 'C Level Job?': 0.0, 'Primary Affiliation': 1.0, 'Next Reunion': 1.0, 'RFM Score': '2', 'Honor': 0.0, 'Hard Credit': 0.0, 'Soft Credit': 0.0, 'Total Commitment': 0.0, 'Total Commitment Bin': 0.0}}, {'43.0': {'#GiftsL5Y': 1.0, '#GiftsL6-10Y': 0.0, '$GivingL10Y': 205.0, 'Total Lifetime Hard Commitment': 205.0, 'Total Lifetime Soft Commitment': 0.0, 'Total Lifetime Commitment': 205.0, '#CommitteesL10Y': 0.0, '#Reunions': 0.0, '#Sports': 0.0, '#Student Activities': 2.0, '?Has Degree?': 1.0, 'Age': 30.0, 'Class Year': 7.0, 'C Level Job?': 0.0, 'Primary Affiliation': 1.0, 'Next Reunion': 1.0, 'RFM Score': '2', 'Honor': 0.0, 'Hard Credit': 0.0, 'Soft Credit': 0.0, 'Total Commitment': 0.0, 'Total Commitment Bin': 0.0}}, {'47.0': {'#GiftsL5Y': 0.0, '#GiftsL6-10Y': 1.0, '$GivingL10Y': 50.0, 'Total Lifetime Hard Commitment': 100.0, 'Total Lifetime Soft Commitment': 0.0, 'Total Lifetime Commitment': 100.0, '#CommitteesL10Y': 1.0, '#Reunions': 0.0, '#Sports': 0.0, '#Student Activities': 1.0, '?Has Degree?': 1.0, 'Age': 30.0, 'Class Year': 7.0, 'C Level Job?': 0.0, 'Primary Affiliation': 1.0, 'Next Reunion': 1.0, 'RFM Score': '3', 'Honor': 0.0, 'Hard Credit': 0.0, 'Soft Credit': 0.0, 'Total Commitment': 0.0, 'Total Commitment Bin': 0.0}}, {'48.0': {'#GiftsL5Y': 0.0, '#GiftsL6-10Y': 0.0, '$GivingL10Y': 0.0, 'Total Lifetime Hard Commitment': 83.33, 'Total Lifetime Soft Commitment': 0.0, 'Total Lifetime Commitment': 83.33, '#CommitteesL10Y': 0.0, '#Reunions': 0.0, '#Sports': 0.0, '#Student Activities': 0.0, '?Has Degree?': 1.0, 'Age': 30.0, 'Class Year': 7.0, 'C Level Job?': 0.0, 'Primary Affiliation': 1.0, 'Next Reunion': 1.0, 'RFM Score': '3', 'Honor': 0.0, 'Hard Credit': 0.0, 'Soft Credit': 0.0, 'Total Commitment': 0.0, 'Total Commitment Bin': 0.0}}, {'54.0': {'#GiftsL5Y': 1.0, '#GiftsL6-10Y': 1.0, '$GivingL10Y': 85.0, 'Total Lifetime Hard Commitment': 105.0, 'Total Lifetime Soft Commitment': 0.0, 'Total Lifetime Commitment': 105.0, '#CommitteesL10Y': 0.0, '#Reunions': 1.0, '#Sports': 0.0, '#Student Activities': 1.0, '?Has Degree?': 1.0, 'Age': 30.0, 'Class Year': 7.0, 'C Level Job?': 0.0, 'Primary Affiliation': 1.0, 'Next Reunion': 1.0, 'RFM Score': '2', 'Honor': 0.0, 'Hard Credit': 0.0, 'Soft Credit': 0.0, 'Total Commitment': 0.0, 'Total Commitment Bin': 0.0}}, {'56.0': {'#GiftsL5Y': 1.0, '#GiftsL6-10Y': 0.0, '$GivingL10Y': 135.0, 'Total Lifetime Hard Commitment': 135.0, 'Total Lifetime Soft Commitment': 0.0, 'Total Lifetime Commitment': 135.0, '#CommitteesL10Y': 0.0, '#Reunions': 0.0, '#Sports': 1.0, '#Student Activities': 1.0, '?Has Degree?': 1.0, 'Age': 30.0, 'Class Year': 7.0, 'C Level Job?': 0.0, 'Primary Affiliation': 1.0, 'Next Reunion': 1.0, 'RFM Score': '2', 'Honor': 0.0, 'Hard Credit': 0.0, 'Soft Credit': 0.0, 'Total Commitment': 0.0, 'Total Commitment Bin': 0.0}}, {'58.0': {'#GiftsL5Y': 1.0, '#GiftsL6-10Y': 0.0, '$GivingL10Y': 100.0, 'Total Lifetime Hard Commitment': 100.0, 'Total Lifetime Soft Commitment': 0.0, 'Total Lifetime Commitment': 100.0, '#CommitteesL10Y': 1.0, '#Reunions': 0.0, '#Sports': 0.0, '#Student Activities': 0.0, '?Has Degree?': 1.0, 'Age': 30.0, 'Class Year': 7.0, 'C Level Job?': 0.0, 'Primary Affiliation': 1.0, 'Next Reunion': 1.0, 'RFM Score': '2', 'Honor': 0.0, 'Hard Credit': 0.0, 'Soft Credit': 0.0, 'Total Commitment': 0.0, 'Total Commitment Bin': 0.0}}]
categories = ['#GiftsL5Y', '#GiftsL6-10Y', '$GivingL10Y', 'Total Lifetime Hard Commitment', 'Total Lifetime Soft Commitment', 'Total Lifetime Commitment', '#CommitteesL10Y', '#Reunions', '#Sports', '#Student Activities', '?Has Degree?', 'Age', 'Class Year', 'C Level Job?', 'Primary Affiliation', 'Next Reunion', 'RFM Score', 'Honor', 'Hard Credit', 'Soft Credit', 'Total Commitment', 'Total Commitment Bin']
ids = []
for donor in donors:
    ids.append(list(donor.keys()))
for i in ids:
    string = ' '.join([str(item) for item in i])
    ids[ids.index(i)] = string

# Initialize matrix A
A = np.zeros((1, 22))
# Add data for each donor corresponding to each row and category in matrix A
for donor in donors:
    arr = np.zeros((1, 22))
    idx = donors.index(donor)
    for category in categories[:-1]:
        arr[0][categories.index(category)] = donor[ids[idx]][category]
    arr[0][-1] = 1.0
    A = np.vstack((A, arr))
A = np.delete(A, 0, axis=0)
noise = np.random.rand(A.shape[0], A.shape[1]) / 100
A += noise

coefficients = [
    [-3.08677854e-01, 1.50412814e-02, 5.32748480e-02, 7.91524882e-02, 8.02049325e-02, 2.76799406e-02, 2.24194464e-02,
     2.23274524e-02, 6.82145528e-03, 1.81417322e-03, 4.68051425e-04, -8.28494685e-04],
    [1.12146349e-01, -1.34123639e-03, -1.49924703e-03, -1.01676906e-02, -5.25105206e-02, -2.05647691e-02,
     -7.31274239e-03, -6.55434559e-03, -1.94523279e-03, -3.35286494e-03, -3.34957928e-03, -3.71210817e-03],
    [2.79385037e-08, 6.28367056e-09, -1.76121873e-08, -3.54937387e-08, -6.44082838e-08, -2.82535228e-08,
     -2.18549095e-08, -1.42377500e-08, 1.39492653e-09, 1.16101904e-08, 2.98440008e-08, 1.03295033e-07],
    [1.85050220e-01, -4.96429693e-02, 1.19000447e-01, 7.57658401e-02, -2.00959283e-01, -1.57869216e-01, -1.46010192e-02,
     1.15809237e-01, 1.15411693e-01, -4.92818155e-02, -1.19795376e-01, -7.48510463e-03],
    [1.85050122e-01, -4.96429838e-02, 1.19000436e-01, 7.57658262e-02, -2.00959306e-01, -1.57869219e-01, -1.46010229e-02,
     1.15809483e-01, 1.15411690e-01, -4.92818222e-02, -1.19795408e-01, -7.48514812e-03],
    [-1.85050189e-01, 4.96429691e-02, -1.19000443e-01, -7.57658322e-02, 2.00959293e-01, 1.57869219e-01, 1.46010245e-02,
     -1.15809236e-01, -1.15411694e-01, 4.92818147e-02, 1.19795376e-01, 7.48504545e-03],
    [8.01554566e-05, -3.26373508e-03, -9.05938561e-03, -1.93292245e-02, -2.45684740e-02, -8.51091189e-03,
     5.59819610e-03, 1.79297046e-02, 8.56878918e-03, 1.12400419e-02, 1.08175789e-02, 1.07259344e-02],
    [-1.50575116e-02, -2.81744330e-03, -7.41112296e-03, -8.88936253e-03, 8.03008110e-03, 6.72911281e-03, 6.48864509e-03,
     6.09654293e-03, 1.63684259e-03, 2.72607675e-03, 1.28714550e-03, 1.17143080e-03],
    [7.86054118e-04, -5.08682982e-03, -1.55578232e-02, -4.87590270e-03, -1.03771965e-02, 4.67397474e-03, 7.08754868e-03,
     1.59587836e-02, 2.28579736e-03, 3.03962317e-03, 5.58412829e-04, 1.62353016e-03],
    [3.09720734e-03, -3.61865433e-03, -5.36550905e-03, -5.79571155e-03, -4.32960709e-03, 6.58459757e-03, 2.67859129e-03,
     2.26671558e-03, 1.19799672e-03, 1.56198489e-03, 1.05601536e-03, 6.44014958e-04],
    [2.79591351e-02, -1.71607028e-02, -3.38325633e-02, -6.16628928e-03, -5.03828778e-02, 2.13354522e-02, 1.88456904e-02,
     2.59519087e-02, 5.74475050e-03, 4.27441023e-03, 2.83842776e-03, -5.27976377e-05],
    [-3.42525727e-03, 5.11325032e-04, 6.83012450e-04, 1.49883424e-03, 3.13869067e-03, 2.91563661e-04, -5.16571415e-04,
     -1.06874566e-03, -3.49357973e-04, -2.36826225e-04, -3.20589266e-04, -3.10452363e-04],
    [4.77068814e-03, -8.64442398e-04, -1.56569564e-03, -2.75982465e-03, -4.73912319e-03, 2.00045061e-04, 9.78436886e-04,
     1.97889807e-03, 5.48406751e-04, 5.01486581e-04, 5.43186946e-04, 5.04975336e-04],
    [1.41418466e-02, -1.35365155e-02, -2.66895542e-02, -5.05348984e-02, -4.38403497e-02, 9.19395390e-03, 2.57170803e-02,
     3.75898334e-02, 1.59343566e-02, 1.34835292e-02, 8.31343896e-03, 9.66323454e-03],
    [1.14240065e-02, 1.35409155e-02, 2.45274238e-02, 2.64410705e-03, 2.95036021e-02, -2.02400334e-02, -1.69237436e-02,
     -2.78514838e-02, 6.95209682e-04, -7.21958818e-03, -6.66878749e-03, -2.49169536e-03],
    [-4.84984060e-02, 1.44717096e-02, 3.22978683e-02, 3.53713100e-02, 3.60924149e-02, -7.46454831e-03, -1.59119213e-02,
     -1.73959347e-02, -1.49950118e-02, -6.41238344e-03, -2.12580829e-03, -5.38376200e-03],
    [3.25322843e-01, -5.07499038e-03, 1.21167774e-02, -2.75122981e-02, -1.56606766e-01, -6.52709890e-02,
     -2.97830515e-02, -1.71848301e-02, -7.51543508e-03, -1.03225213e-02, -9.16804760e-03, -1.01739966e-02],
    [1.78167803e-03, 5.81493442e-04, 3.97625421e-04, -3.29044998e-03, -1.22266789e-02, 9.23480848e-05, 5.30009778e-04,
     4.78292140e-03, 2.54095506e-03, 9.71103766e-04, 9.46952265e-04, 3.19861597e-03],
    [1.01238821e-01, 1.38634836e-01, -7.48204630e-01, 6.62569446e-02, -4.17876870e-01, -1.87796398e-01, -2.68312269e-01,
     6.18059848e-01, 1.69583506e-01, 3.20968955e-01, 8.54740367e-02, 8.70707246e-02],
    [1.01150030e-01, 1.38624235e-01, -7.48237307e-01, 6.61973609e-02, -4.17970852e-01, -1.87803311e-01, -2.68286830e-01,
     6.18137102e-01, 1.69653074e-01, 3.21059068e-01, 8.55072506e-02, 8.70677133e-02],
    [-1.01239698e-01, -1.38634901e-01, 7.48204543e-01, -6.62571175e-02, 4.17876548e-01, 1.87796274e-01, 2.68312175e-01,
     -6.18059887e-01, -1.69583474e-01, -3.20968884e-01, -8.54739128e-02, -8.70691710e-02],
    [9.22055677e-02, -9.79299044e-01, -1.00958631e+00, -9.10588662e-01, -5.34672877e-01, -8.36125661e-01,
     -9.18786223e-01, -9.62341918e-01, -9.78698053e-01, -9.72263705e-01, -9.69335617e-01, -9.65238050e-01]]

# Build coefficients into matrix
x = np.zeros((22, 12))

for feature in coefficients:
    for val in feature:
        x[coefficients.index(feature)][feature.index(val)] = val

for i in range(0, 12):
    x[21][i] = 1

probabilities = A@x

predictions = []
for row in probabilities:
    predicted = np.argmax(list(row))
    predictions.append(predicted)

print(predictions)
