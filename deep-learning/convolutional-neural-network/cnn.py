import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print(tf.__version__)

# processing the train set
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
training_set = train_datagen.flow_from_directory('dataset/training_set', target_size=(64, 64), batch_size=32, class_mode='binary')

# processing the test set
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory('dataset/test_set', target_size=(64, 64), batch_size=32, class_mode='binary')

# build CNNNNNNNNNNNNNNNNNNNNNNNN
## init cnn
cnn = tf.keras.models.Sequential()

## step 1 - convolution layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64,64,3]))

## step 2 - pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

## adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

## step 3 - flattening
cnn.add(tf.keras.layers.Flatten())

## step 4 - full connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

## step 5 - output layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# train CNNNNNNNNNNNNNNNNNNN
## compile cnn
cnn.compile(optimizer = 'adam', loss='binary_crossentropy', metrics=['accuracy'])

## train the cnn on training set and evaluating it on the test set
cnn.fit(x=training_set, validation_data=test_set, epochs = 25)

# making a single prediction
import numpy as np
from tensorflow.keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/dog.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = cnn.predict(test_image)
training_set.class_indices
if (result[0][0] == 1):
    prediction = 'dog'
else:
    prediction = 'cat'

print(prediction)