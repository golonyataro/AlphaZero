#alpha-2

#パッケージのインポート
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.layers import Activation, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#データセットの準備
(train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()

#データセットのシェイプの確認
print(train_data.shape)
print(train_labels.shape)
print(test_data.shape)
print(test_labels.shape)

#データセットのデータの確認
# column_names = ['CRIN', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PIRATIO', 'B', 'LSTAT']
# df = pd.DataFrame(train_data, columns=column_names)
# df.head()

#データセットのラベルの確認
print(train_labels[0:10])

#データセットのシャッフルの前確認
order = np.argsort(np.random.random(train_labels.shape))
train_data = train_data[order]
train_labels = train_labels[order]

#データセットの正規化の前処理
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std

#モデルの作成
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(13,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

#コンパイル
model.compile(loss='mse', optimizer=Adam(lr=0.001), metrics=['mae'])

#EarlyStoppingの準備
early_stop = EarlyStopping(monitor='val_loss', patience=20)

#学習
history = model.fit(train_data, train_labels, epochs=500, validation_split=0.2, callbacks=[early_stop])

#評価
test_loss, test_mae = model.evaluate(test_data, test_labels)
print('loss:{:.3f}\nmae: {:.3f}'.format(test_loss, test_mae))

#推論する値段の表示
print(np.round(test_labels[0:10]))

#推論した値段の表示
test_predictions = model.predict(test_data[0:10]).flatten()
print(np.round(test_predictions))

#グラフの表示
plt.plot(history.history['mean_absolute_error'], label="train mae")
plt.plot(history.history['val_mean_absolute_error'], label='val mae')
plt.xlabel('epoch')
plt.ylabel('mae [1000$]')
plt.legend(loc='best')
plt.ylim([0,5])
plt.show()