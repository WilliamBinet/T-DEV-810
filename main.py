import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten



if __name__ == '__main__':
    print(tf.config.experimental.list_physical_devices('GPU'))
    classifier = Sequential()
    classifier.add(Conv2D(24, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Conv2D(24, (3, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Conv2D(24, (3, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Flatten())
    classifier.add(Dense(units = 128, activation = 'relu'))
    classifier.add(Dense(units = 1, activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


    train_datagen = ImageDataGenerator(rescale = 1./255,
                                       shear_range = 0.2,
                                       zoom_range = 0.2,
                                       horizontal_flip = True)

    test_datagen = ImageDataGenerator(rescale = 1./255)

    training_set = train_datagen.flow_from_directory('chest_xray/train',
                                                     target_size = (64, 64),
                                                     batch_size = 24,
                                                     class_mode = 'binary')

    test_set = test_datagen.flow_from_directory('chest_xray/test',
                                                target_size = (64, 64),
                                                batch_size = 24,
                                                class_mode = 'binary')

    classifier.fit_generator(training_set,
                             epochs = 100,
                             validation_data = test_set)

    test_accu = classifier.evaluate_generator(test_set)

    print('The testing accuracy is :', test_accu[1] * 100, '%')

    print('The testing accuracy is :', test_accu[1] * 100, '%')

