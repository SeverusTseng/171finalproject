ConvertData(): convert .npy to .csv and save(only need to run once)
LoadData(): load data from converted .csv
NAPercent(train_x): compute NA percentage and save
deleteNA(train_x, test, NA): delete features with NA>5% in training data and corresponding testing data
DeleteDuplicate(train_x, test): delete duplicate features(pairs with over 30000 same number are regarded as duplicate) in training data and corresponding testing data
GoldenFeature(train_x, test): find pairs with correlation bigger than 0.996 and keep their difference
FinalFeature(train_x, test): remove one of pair features with correlation bigger than 0.99 in remaing data

Output: original train_x.csv, train_y.csv, test.csv; 
And data after selection: FinalFeature.csv(for training); FinalTest.csv(for test);

To read data: 
train_x = np.loadtxt(open("FinalFeature.csv","rb"), delimiter=",", skiprows=0)

train_x will be a np.array, train_x[i][j] is the (j+1)th feature for (i+1)th id
