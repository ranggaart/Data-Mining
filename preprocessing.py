import numpy as np
import matplotlib.pyplot as  plt
import pandas as pd 

dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print('\nRangga Aristianto')
print('A11.2022.14568')
print('\nData Kasus:')
print('\nData X:')
print(x)
print('\nData Y:')
print(y)

# Menghilangkan Missing Value (nan)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy='mean')
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])
print('\n\nData Setelah Transformasi (Mean):')
print(x)

# Encoding Data Kategori (Atribut)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer (transformers=[('encoder', OneHotEncoder (), [0])], remainder= 'passthrough')
x = np. array(ct.fit_transform(x))

print ('\nData Kategori Atribut:')
print(x)

# Encoding Data Kategori (Class/Label)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder ()
y = le.fit_transform (y)

print ('\n\nData Kategori Class/Label:')
print(y)

# Membagi Dataset ke dalam Training Set dan Test Set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)

print ('\n\nData Training Set dan Test Set:')
print('\nData X Train:')
print(x_train)
print('\nData Y Train:')
print(y_train)

print('\nData X Test:')
print(x_test)
print('\nData y Test:')
print(y_test)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])
x_test[:, 3:] = sc.transform(x_test[:, 3:])

print ('\n\nData Feature Scaling:')
print('\nData X Train:')
print(x_train)
print('\nData X Test:')
print(x_test)