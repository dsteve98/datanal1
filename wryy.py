import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import r2_score
from keras import optimizers

#Variables
dataset=np.loadtxt("training-data.csv", delimiter=",")
x=dataset[:,0:26]
y=dataset[:,26] #target 1
z=dataset[:,27] #target 2
y=np.reshape(y, (-1,1))
z=np.reshape(z, (-1,1))
scaler = MinMaxScaler()
print(scaler.fit(x))
print(scaler.fit(y))
print(scaler.fit(z))
xscale=scaler.transform(x)
yscale=scaler.transform(y)
yscale=scaler.transform(z)

X_train, X_test, y_train, y_test = train_test_split(xscale, yscale)


model = Sequential()
model.add(Dense(36, input_dim=26, kernel_initializer='normal', activation='relu'))
model.add(Dense(26, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()

adam = optimizers.Adam(lr=0.001)
model.compile(loss='mse', optimizer=adam, metrics=['mse','mae'])

history = model.fit(X_train, y_train, epochs=500, batch_size=32,  verbose=1, validation_split=0.2)

print(history.history.keys())
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# Runs model (the one with the activation function, although this doesn't really matter as they perform the same)
# with its current weights on the training and testing data
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculates and prints r2 score of training and testing data
print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(y_train, y_train_pred)))
print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test, y_test_pred)))