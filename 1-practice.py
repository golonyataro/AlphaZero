# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.layers import Activation, Dense
from keras.models import Sequential, load_model
from keras.utils.np_utils import to_categorical

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 784)[:6000]
X_test = X_test.reshape(X_test.shape[0], 784)[:1000]
y_train = to_categorical(y_train)[:6000]
y_test = to_categorical(y_test)[:1000]

model = Sequential()
model.add(Dense(256, input_dim=784))
model.add(Activation("sigmoid"))
model.add(Dense(128))
model.add(Activation("sigmoid"))
model.add(Dense(10))
model.add(Activation("softmax"))

model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=1)
# verbose に １ を指定した場合は学習の進捗度合いを出力し、 0 の場合は出力しません

score = model.evaluate(X_test, y_test, verbose=False)
print("evaluate loss: {0[0]}\nevaluate acc: {0[1]}".format(score))

# X_testの最初の10枚の予測されたラベルを表示しましょう
#---------------------------
pred = np.argmax(model.predict(X_test[0:10]), axis=1)
print(pred)
#---------------------------

# テストデータの最初の10枚を表示します
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(X_test[i].reshape((28,28)), "gray")
plt.show()