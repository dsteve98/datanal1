import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore

file = r'training-data.xlsx'
df = pd.read_excel(file)

df.drop(['Appliances'], axis = 1, inplace = True)
df.drop(['lights'], axis = 1, inplace = True)

#print(df.head(5))

print(df.dtypes)

print(df.describe())

df = df.replace('?',np.NaN)
df = df.replace('',np.NaN)
df = df.replace(' ',np.NaN)
df = df.replace('-',np.NaN)
df = df.replace('_',np.NaN)

#print(df.isnull().sum())
for col in df.columns:
    print('\t%s: %d' % (col,df[col].isnull().sum()))

Z = df.apply(zscore)
Z.boxplot()
plt.show()