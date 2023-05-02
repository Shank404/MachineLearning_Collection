from keras import Input, Model
from keras.applications import InceptionResNetV2
from keras.callbacks import EarlyStopping
from keras.layers import GlobalAveragePooling2D, Dropout, Dense
from matplotlib import pyplot as plt
import keras
import DataAugmenter

# Aufbereitung der Daten (Inkl. Augmentierung)
training_set, validation_set, prediction_set = DataAugmenter.prepareData()

# Anzahl möglicher Klassifizierungen
class_count = len(training_set.class_indices)

# Resnet laden und mit vorgelernten Gewichten bestücken
resNet_Model = InceptionResNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(75, 75, 3)
)

# Friert Gewichte des ResNet ein.
resNet_Model.trainable = False

# Konsolen-Ausgabe der Schichten und Parameter
resNet_Model.summary()

# Definition des zweiten Models (Transfer Learning)
input1 = Input((1, 1, 1536))
globalAveragePooling2D = GlobalAveragePooling2D()(input1)
dropout = Dropout(0.6)(globalAveragePooling2D)
output1 = Dense(class_count, activation='softmax')(dropout)

# Erstellen des Top-Models
topModel = Model(input1, output1)

# Konsolen-Ausgabe der Schichten und Parameter
topModel.summary()

# Speichert den Output der letzten Schicht des VGG16
conv_7b_ac = resNet_Model.get_layer('conv_7b_ac').output

# Verknüpfung zwischen der letzten Layer (ResNet) und ersten Layer (TopModel)
full_output = topModel(conv_7b_ac)

# Erstellen des neuen verknüpften Models
full_model = Model(resNet_Model.input, full_output)

# Kompilieren des Modells
full_model.compile(
    optimizer="sgd",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Stoppt das Training unter gewissen Bedingungen
callback = EarlyStopping(
    monitor="val_accuracy",                 # Zu beobachtender Wert
    mode="max",                             # Stoppt sobald der Wert nicht mehr wächst
    patience=5,                             # Nach 5 Epochen ohne Veränderung wird gestoppt
    restore_best_weights=True               # Nehme die Gewichte mit den besten Ergebnissen
)

# Trainiere das Modell
history = full_model.fit(
    training_set,
    validation_data=validation_set,
    epochs=5,
    callbacks=[callback],
)

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

# Prediction auf das Prediction_Set ausführen.
predictions_val = full_model.predict(prediction_set)
print(predictions_val)


