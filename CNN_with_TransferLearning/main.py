from keras import Input, Model
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.callbacks import EarlyStopping
from keras.layers import Flatten, Dense
from keras.utils import to_categorical
from keras.utils.image_utils import smart_resize
from matplotlib import pyplot as plt
from tensorflow_datasets import load

# Laden eines vordefinierten Datensatzes
(x_Train, y_Train), (x_Test, y_Test), (x_Validation, y_Validation)= load(
    "tf_flowers",
    split=["train[:50%]", "train[:30%]", "train[:20%]"],    # Aufteilung der Trainings- und Testdaten
    batch_size=-1,                                          # Gibt uns einen Tensor zurück!
    as_supervised=True                                      # Datenstruktur wird ein Tupel sein in der Form (Input, Label)
)

# Bilder auf gewünschtes Input-Format skalieren
x_Train = smart_resize(x_Train, (150, 150))
x_Test = smart_resize(x_Test, (150, 150))
x_Validation = smart_resize(x_Validation, (150, 150))
print()

# One-Hot-Encoding
y_Train = to_categorical(y_Train, num_classes=5)
y_Test = to_categorical(y_Test, num_classes=5)
y_Validation = to_categorical(y_Validation, num_classes=5)

# Laden des VGG16 Models
vgg16 = VGG16(
    weights="imagenet",                                     # Vorgelernte Gewichte verwenden
    include_top=False,                                      # Weglassen der letzten Schicht
    input_shape=x_Train[0].shape                            # Definition der Input-Dimensionen
)

# Einfrieren der Gewichte
vgg16.trainable = False

# Vorverarbeiten der Daten für VGG16 (Normalisierung, RGB -> BGR)
x_Train = preprocess_input(x_Train)
x_Test = preprocess_input(x_Test)

# Konsolen-Ausgabe der Schichten und Parameter
vgg16.summary()

# Definition des zweiten Models (Transfer Learning)
input1 = Input((4, 4, 512))
flatten = Flatten()(input1)
dense1 = Dense(50, activation='relu')(flatten)
dense2 = Dense(20, activation='relu')(dense1)
output1 = Dense(5, activation='softmax')(dense2)

# Erstellen des Top-Models
topModel = Model(input1, output1)

# Konsolen-Ausgabe der Schichten und Parameter
topModel.summary()

# Speichert den Output der letzten Schicht des VGG16
block5_pool = vgg16.get_layer('block5_pool').output

# Verknüpfung zwischen der letzten Layer (VGG16) und ersten Layer (TopModel)
full_output = topModel(block5_pool)

# Erstellen des neuen verknüpften Models
full_model = Model(vgg16.input, full_output)

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
    x_Train,
    y_Train,
    validation_data=(x_Validation, y_Validation),
    epochs=50,
    batch_size=32,
    callbacks=[callback],
)

# Evaluiere das Modell mit der Testmenge
full_model.evaluate(x_Test, y_Test)

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