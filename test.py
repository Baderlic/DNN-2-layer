import numpy as np
import pickle

A = np.load('mnist.npz')
x1,x2 = A['x_train'],A['x_test']
y1,y2 = A['y_train'],A['y_test']
nx2 = np.reshape(x2, (-1, 784))
y2 = y2.reshape(y2.shape[0],1)

with open('model.pickle','rb') as file:
    m = pickle.load(file)

loss_test = m.lost_func(nx2, y2)
accuracy_test = m.predict(nx2,y2)
print("The accuracy of selected model on test data is", accuracy_test, "and loss is", loss_test)