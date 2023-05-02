from keras import Input, Model
from keras.layers import Conv2D, BatchNormalization, ReLU, MaxPooling2D, Dense, Softmax, Flatten, Dropout
from keras.losses import CategoricalCrossentropy
from matplotlib import pyplot as plt

import DataAugmenter

# Hinweis strukturierte und unstrukturierte Daten!
# Strukturiert      -> Tabelle                      <-  Flaches NN
# Unstrukturiert    -> Bilder, Audio, Video         <-  Deep NN

# Aufbereitung der Daten (Inkl. Augmentierung)
training_set, testing_set, validation_set = DataAugmenter.prepareData()

# Stellt das erste Bild aus fünf Batches dar.
DataAugmenter.showSamples(training_set)

# Definition der CNN-Schichten
inputLayer = Input(shape=(28, 28, 1))

x = Conv2D(8, kernel_size=(3, 3), padding="same")(inputLayer)
x = BatchNormalization()(x)
x = ReLU()(x)                                # BAD REGEL - S.56 Generative - Batch, Aktivierung, Dropout
x = Dropout(0.2)(x)
x = MaxPooling2D()(x)

x = Conv2D(16, kernel_size=(3, 3), padding="same")(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = Dropout(0.2)(x)
x = MaxPooling2D()(x)

x = Conv2D(32, kernel_size=(3, 3), padding="same")(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = Dropout(0.2)(x)

x = Flatten()(x)
output = Dense(10, activation="softmax")(x)

model = Model(inputs=inputLayer, outputs=output)

# Print-Ausgabe des CNN-Modells
print(model.summary())

# Konfiguriert das Modell fürs Training
model.compile(optimizer='sgd', loss=CategoricalCrossentropy(), metrics=['accuracy'])

# Trainiert das Modell
history = model.fit(training_set, epochs=20, validation_data=validation_set, validation_freq=2, shuffle='true')

# Evaluiert die Testmenge
print("\n\nEvaluiere die Testmenge")
model.evaluate(testing_set)

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