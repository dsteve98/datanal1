import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file = r'training-data.xlsx'
df = pd.read_excel(file)

print(df.head(5))

print(df.dtypes)

df.boxplot()
plt.show()