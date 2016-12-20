from linearRegression import mean
from linearRegression import covariance
from linearRegression import variance
from linearRegression import coefficients
from linearRegression import regress

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

house_price = pd.read_csv("C:\Users/aalla/Documents/cscc11/housing/train.csv") 
house_id= house_price.Id
house_test = pd.read_csv("C:\Users/aalla/Documents/cscc11/housing/test.csv")
house_test_id = house_test.Id

# Make a new dataframe containing only numeric data columns

numeric_feature = [a for a in range(len(house_price.dtypes)) if house_price.dtypes[a] in ['int64','float64']]
numeric_fet_test = [a for a in range(len(house_test.dtypes)) if house_test.dtypes[a] in ['int64','float64']]
numeric_data = house_price.iloc[:,numeric_feature]
numeric_test = house_test.iloc[:,numeric_fet_test]

# Make different data frame for the category data in train and test
cat_name = house_price.columns.difference(house_price.columns[numeric_feature])
cat_data = house_price.ix[:,cat_name]
cat_name_test = house_test.columns.difference(house_test.columns[numeric_fet_test])
cat_test = house_price.ix[:,cat_name_test]

# Replace numpy nan value is string NA values
# Remember, there are different ways to deal with missing values but here we are not going to deal with it
new_dataset = house_price.copy()
new_dataset = new_dataset.replace(to_replace=np.nan,value="NA")
new_test = house_test.copy()
new_test = new_test.replace(to_replace=np.nan,value="NA")

labels = []
for i in cat_name:
    train_label = new_dataset[i].unique()
    test_label = new_test[i].unique()
    labels.append(list(set(train_label)|set(test_label)))

    
encoded_cat = []
for i in range(len(cat_name)):
    label_coder = LabelEncoder()
    label_coder.fit(labels[i])
    cat_col = label_coder.transform(new_dataset.loc[:,cat_name[i]])
    cat_col = cat_col.reshape(new_dataset.shape[0],1)
    onehot_encoder = OneHotEncoder(sparse=False,n_values=len(labels[i]))
    cat_col = onehot_encoder.fit_transform(cat_col)
    encoded_cat.append(cat_col)

encoded_frame = np.column_stack(encoded_cat)

# Concatenate encoded category and numeric data
new_data = np.concatenate((encoded_frame,numeric_data),axis=1)
new_data = pd.DataFrame(new_data)

# Create Validation set from training set

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
x_train,x_val,y_train,y_val = train_test_split(new_data.iloc[:,0:310],new_data[311],random_state = 0)

# Fill the missing values of numeric data with the mean of the columns
x_train = x_train.fillna(x_train.mean())
x_val = x_val.fillna(x_val.mean())

regression = regress(x_train,y_train)