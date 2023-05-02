import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

# Laden des MNIST-Datensatzes
from matplotlib import pyplot as plt

(x_Training, y_Training), (x_Testing, y_Testing) = mnist.load_data()

# Definition der Input-Dimensionen
input_shape = (28, 28, 1)

# Datenpunkte als Float darstellen
x_Training = x_Training.astype('float32')
x_Testing = x_Testing.astype('float32')

# Normalisierung
x_Training /= 255
x_Testing /= 255

# One-Hot-Encoding
y_Training = keras.utils.to_categorical(y_Training, 10)
y_Testing = keras.utils.to_categorical(y_Testing, 10)

# Modell-Erstellung
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# Kompilieren des Modells
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

# Trainiere das Modell mit den Trainingsdaten && validiere es mit den Testdaten
history = model.fit(x_Training, y_Training, batch_size=128, epochs=12, verbose=1, validation_data=(x_Testing, y_Testing))

# Plotten der Accuracy und des Losses
# Accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# Loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
