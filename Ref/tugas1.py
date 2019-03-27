import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.model_selection import train_test_split

from sklearn import datasets
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA as sklearnPCA

data = pd.read_csv(
    "adult.data",
    names=[
        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "Target"],
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")

print(data.dtypes)

#obj_df = data.select_dtypes(include=['object']).copy()
print(data["Country"].value_counts())

cleanup_nums = {"Workclass": {"Private": 1, "Self-emp-not-inc": 2, "Local-gov": 3, "State-gov": 4, "Self-emp-inc": 5, "Federal-gov": 6, "Without-pay": 7, "Never-worked": 8},
				"Education": {"HS-grad": 1, "Some-college": 2, "Bachelors": 3, "Masters": 4, "Assoc-voc": 5, "11th": 6, "Assoc-acdm": 7, "10th": 8, "7th-8th": 9, "Prof-school": 10, "9th": 11, "12th": 12, "Doctorate": 13, "5th-6th": 14, "1st-4th": 15, "Preschool": 16},
				"Martial Status": {"Married-civ-spouse": 1, "Never-married": 2, "Divorced": 3, "Separated": 4, "Widowed": 5, "Married-spouse-absent": 6, "Married-AF-spouse": 7},
				"Occupation": {"Prof-specialty": 1, "Craft-repair": 2, "Exec-managerial": 3, "Adm-clerical": 4, "Sales": 5, "Other-service": 6, "Machine-op-inspct": 7, "Transport-moving": 8, "Handlers-cleaners": 9, "Farming-fishing": 10, "Tech-support": 11, "Protective-serv": 12, "Priv-house-serv": 13, "Armed-Forces": 14},
				"Relationship": {"Husband": 1, "Not-in-family": 2, "Own-child": 3, "Unmarried": 4, "Wife": 5, "Other-relative": 6},
				"Race": {"White": 1, "Black": 2, "Asian-Pac-Islander": 3, "Amer-Indian-Eskimo": 4, "Other": 5},
				"Sex": {"Male": 1, "Female": 2},
				"Country": {"United-States": 1, "Mexico": 2, "Philippines": 3, "Germany": 4, "Canada": 5, "Puerto-Rico": 6, "El-Salvador": 7, "India": 8, "Cuba": 9, "Jamaica": 10, "South": 11, "China": 12,
				"Italy": 13, "Dominican-Republic": 14, "Vietnam": 15, "Guatemala": 16, "Japan": 17, "Poland": 18, "Columbia": 19, "Taiwan": 20, "Haiti": 21, "Iran": 22, "Portugal": 23, "Nicaragua": 24, "Peru": 25,
				"France": 26, "Greece": 27, "Ecuador": 28, "Ireland": 29, "Hong": 30, "Trinadad&Tobago": 31, "Cambodia": 32, "Laos": 33, "Thailand": 34, "Yugoslavia": 35, "Outlying-US(Guam-USVI-etc)": 36, "Hungary": 37,
				"Honduras": 38, "Scotland": 39, "Holand-Netherlands": 40, "England": 41},
				"Target": {"<=50K": 1, ">50K": 2}}
data.replace(cleanup_nums, inplace=True)
data = data.astype({"Workclass": object, "Education": object, "Education-Num": object, "Martial Status": object, "Occupation": object, "Relationship": object, "Race": object, "Sex": object, "Country": object, "Target": object})
print(data.dtypes)
print(data.head())
data = data.replace(' ?',np.NaN)
for col in data.columns:
    print('\t%s: %d' % (col,data[col].isnull().sum()))

data = data.dropna()

numeric_cols = data.select_dtypes(include=[np.number]).columns
Z = data[numeric_cols].apply(zscore)
#Z = pd.DataFrame(Z, columns=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o'])
print(Z.head())
Z.boxplot()
plt.show() #comm

print('Number of rows before discarding outliers = %d' % (Z.shape[0]))

#Z2 = Z.loc[(Z['Age'] > -3) & (Z['Age'] <= 3) & (Z['fnlwgt'] <= 2.5) & (Z['Capital Gain'] <= 1) & (Z['Capital Loss'] <= 1) & (Z['Hours per week'] <= 1) & (Z['Hours per week'] > -1) ,:]
Z2 = Z.loc[(Z['Age'] > -3) & (Z.Age <= 3),:]
print('Number of rows after discarding outliers = %d' % (Z2.shape[0]))
Z2 = Z2.astype({"Workclass": object, "Education": object, "Education-Num": object, "Martial Status": object, "Occupation": object, "Relationship": object, "Race": object, "Sex": object, "Country": object, "Target": object})

Z2.boxplot()
plt.show() #comm


print('Number of rows before discarding noise = %d' % (Z.shape[0]))

Z2 = data.loc[(Z['Age'] > -3) & (Z.Age <= 3),:]
print('Number of rows after discarding noise = %d' % (Z2.shape[0]))
Z2 = Z2.astype({"Workclass": object, "Education": object, "Education-Num": object, "Martial Status": object, "Occupation": object, "Relationship": object, "Race": object, "Sex": object, "Country": object, "Target": object})

Z2.boxplot()
plt.show() #comm

dups = Z2.duplicated()
print('Number of duplicate rows = %d' % (dups.sum()))
print('Number of rows before discarding duplicates = %d' % (Z2.shape[0]))
Z2 = Z2.drop_duplicates()
print('Number of rows after discarding duplicates = %d' % (Z2.shape[0]))

data = Z2
data = data.astype({"Workclass": object, "Education": object, "Education-Num": object, "Martial Status": object, "Occupation": object, "Relationship": object, "Race": object, "Sex": object, "Country": object, "Target": object})

for col in data.columns:
    print('\t%s: %d' % (col,data[col].isnull().sum()))

print(data.dtypes)

target=data.iloc[:, -1]
data2 = data.iloc[:,0:-1]
#print("data:")
#print(data2)

pca = sklearnPCA(n_components=2)
#Xstd_pca = pca.fit_transform(X_std)
X_pca = pca.fit_transform(data2)

target[target==1] = 1
target[target==2] = 2

plt.scatter(X_pca[:, 0], X_pca[:, 1], marker='o', c=target)
plt.show()
print(X_pca)


Train, Test = train_test_split(data, test_size=0.30, stratify=data['Target'])
label1a = sum(data['Target']==1)
label1b = sum(Train['Target']==1)
label2a = sum(data['Target']==2)
label2b = sum(Train['Target']==2)
print('Jumlah kelas 1(<=50K) sebenarnya', label1a)
print('Jumlah kelas 2(>50K) stratified sampling', label1b)
print('Jumlah kelas 1 sebenarnya', label2a)
print('Jumlah kelas 2 stratified sampling', label2b)

