from keras_preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


def prepareData():
    training_generator = ImageDataGenerator(
        shear_range=0.1,
        zoom_range=0.15,
        rotation_range=10,
        width_shift_range=0.05,
        horizontal_flip=True,
        validation_split=0.1,
        rescale=1.0/255
    )

    training_set = training_generator.flow_from_directory(
        directory='./images/64',
        target_size=(75, 75),
        batch_size=32,
        color_mode='rgb',
        class_mode='categorical',
        shuffle=True,
        subset='training'
    )

    validation_set = training_generator.flow_from_directory(
        directory='./images/64',
        target_size=(75, 75),
        batch_size=32,
        color_mode='rgb',
        class_mode='categorical',
        shuffle=True,
        subset='validation'
    )

    prediction_set = training_generator.flow_from_directory(
        directory='./images/Prediction',
        target_size=(75, 75),
        color_mode='rgb',
        class_mode='categorical',
        shuffle=True
    )

    return training_set, validation_set, prediction_set


def showSamples(data):
    fig = plt.figure()
    for i in range(9):
        img, label = data.next()
        plot = fig.add_subplot(3, 3, i + 1)
        print(f"Image {i} : {img.shape}")
        plot.imshow(img[i], cmap='gray', interpolation='none')
    plt.tight_layout()
