#alpha-3

#パッケージのインポート
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Activation, Dense, Dropout, Conv2D, Flatten, MaxPool2D
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

#データセットの準備
(train_images, train_labels),(test_images, test_labels) = cifar10.load_data()

#データセットのシェイプの確認
print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)

# データセットの画像の確認
# for i in range(10):
#   plt.subplot(2, 5, i+1)
#   plt.imshow(train_images[i])
# plt.show()

#データセットのラベルの確認
print(train_labels[0:10])

#データセットの画像の前処理
train_images = train_images.astype('float32')/255.0
test_images = test_images.astype('float32')/255.0

#データセットの画像の前処理後のシェイプの確認
#画像配列の要素である画像は、1次元ではなく、3次元であることに注意
print(train_images.shape)
print(test_images.shape)

#データセットのラベルの前処理
train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

#データセットのラベルの前処理後のシェイプの確認
print(train_labels.shape)
print(test_labels.shape)

#モデルの作成
model = Sequential()

# Conv→Conv→Pool→Dropout
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0, 25))

# Conv→Conv→Pool→Dropout
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0, 25))

# Flatten→Dense→Dropout→Dence
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

#コンパイル
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['acc'])

#学習
history = model.fit(train_images, train_labels, batch_size=128, epochs=20, validation_split=0.1)

#モデルの保存
model.save('convolution.h5')

#モデルの再読み込み
model = load_model('convolution.h5')

#グラフの表示
plt.plot(history.history['acc'], label='acc')
plt.plot(history.history['val_acc'], label='val_acc')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(loc='best')
plt.show()

#評価
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('loss:{:.3f}\nacc: {:.3f}'.format(test_loss, test_acc))

#推論する画像の表示
for i in range(10):
  plt.subplot(2, 5, i+1)
  plt.imshow(test_images[i])
plt.show()

#推論したラベルの表示
test_predictions = model.predict(test_images[0:10])
test_predictions = np.argmax(test_predictions, axis=1)
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
print([labels[n] for n in test_predictions])