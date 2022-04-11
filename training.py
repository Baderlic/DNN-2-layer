import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import model

A = np.load('mnist.npz')
x1,x2 = A['x_train'],A['x_test']
y1,y2 = A['y_train'],A['y_test']
nx1 = np.reshape(x1, (-1, 784))
nx2 = np.reshape(x2, (-1, 784))

X_train,X_val, y_train, y_val  = train_test_split(nx1, y1,test_size=0.25, random_state=50)

y_train = y_train.reshape(y_train.shape[0],1)
y_val = y_val.reshape(y_val.shape[0],1)

batch_num = 2250
batch_size = 20

x_batches = []
y_batches = []
for i in range(batch_num):
    x = X_train[i*batch_size:(i+1)*batch_size, :]
    y = y_train[i*batch_size:(i+1)*batch_size]
    x_batches.append(x)
    y_batches.append(y)

learning_rates = [0.003, 0.001, 0.0005]
reg_lambdas = [2e-4, 1e-4, 5e-5]
hdims = [20, 40, 60]

models = {}

def train(learning_rate, reg_lambda, hdim):
    
    loss_trs = []
    loss_vs = []
    acc_trs = []
    acc_vs = []
    modeling = model.Network(X_train, y_train, learning_rate, reg_lambda, hdim)

    for epoch in range(50):

        loss_tr = modeling.lost_func(X_train, y_train)
        accuracy_tr = modeling.predict(X_train,y_train)

        loss_v = modeling.lost_func(X_val, y_val)
        accuracy_v = modeling.predict(X_val,y_val)

        print('epoch--',epoch)
        print('train--accuracy=',accuracy_tr,'loss=',loss_tr)
        print('valid--accuracy=',accuracy_v,'loss=',loss_v)

        loss_trs.append(loss_tr)
        loss_vs.append(loss_v)

        acc_trs.append(accuracy_tr)
        acc_vs.append(accuracy_v)

        for i in range(batch_num):
            x0,y0 = x_batches[i],y_batches[i]
            modeling.sgd_step(x0,y0)

        if i % 5 == 0:
            modeling.learning_rate *= 0.5

    m = modeling

    return loss_trs,loss_vs,acc_trs,acc_vs,m


# train(0.0005,1e-5,40)
#search the best parameters
accs = []
i = 0
for learning_rate in learning_rates:
    for reg_lambda in reg_lambdas:
        for hdim in hdims:
            i += 1
            loss_trs,loss_vs,acc_trs,acc_vs,m = train(learning_rate,reg_lambda,hdim)
            print('End training. Accuracy=',acc_vs[-1], "i=", i)
            accs.append(acc_vs[-1])

# selected model
train_loss, val_loss, train_acc, val_acc, best_model = train(0.003, 5e-5, 60)
best_model.save()

train_loss_2, val_loss2, train_acc_2, val_acc_2, normal_model = train(0.001, 2e-4, 20)
#visualization
#the best model
# plt.subplots(1,2)
plt.subplot(121)
plt.plot(train_loss, color="c", label="training")
plt.plot(val_loss, color="r", label="validation")
plt.legend()
plt.title("loss")

plt.subplot(122)
plt.plot(train_acc, color="c", label="training")
plt.plot(val_acc, color="r", label="validation")
plt.legend()
plt.title("accuracy")
plt.suptitle("the best model's training")

#the normal model
plt.subplot(121)
plt.plot(train_loss_2, color="c", label="training")
plt.plot(val_loss2, color="r", label="validation")
plt.legend()
plt.title("loss")

plt.subplot(122)
plt.plot(train_acc_2, color="c", label="training")
plt.plot(val_acc_2, color="r", label="validation")
plt.legend()
plt.title("accuracy")
plt.suptitle("the normal model's training")

plt.show()

