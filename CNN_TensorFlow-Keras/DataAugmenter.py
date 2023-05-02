from keras_preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


def prepareData():
    training_generator = ImageDataGenerator(
        shear_range=0.1,  #
        zoom_range=0.15,  #
        rotation_range=10,
        width_shift_range=0.05,
        horizontal_flip=True
    )

    validation_generator = ImageDataGenerator(
        validation_split=0.3
    )

    training_set = training_generator.flow_from_directory(
        directory='images/Train',
        batch_size=16,
        target_size=(28, 28),
        color_mode='grayscale',
        class_mode='categorical',
        shuffle=True
    )

    validation_set = validation_generator.flow_from_directory(
        directory='images/Full',
        batch_size=16,
        target_size=(28, 28),
        color_mode='grayscale',
        class_mode='categorical',
        subset='validation',
        shuffle=True
    )

    testing_set = validation_generator.flow_from_directory(
        directory='images/Full',
        batch_size=16,
        target_size=(28, 28),
        color_mode='grayscale',
        class_mode='categorical',
        shuffle=True,
        subset='training'
    )
    return training_set, testing_set, validation_set


def showSamples(data):
    fig = plt.figure()
    for i in range(9):
        img, label = data.next()
        plot = fig.add_subplot(3, 3, i + 1)
        print(f"Image {i} : {img.shape}")
        plot.imshow(img[i], cmap='gray', interpolation='none')
    plt.tight_layout()
