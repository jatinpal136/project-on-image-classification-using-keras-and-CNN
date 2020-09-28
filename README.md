# project-on-image-classification-using-keras-and-CNN with CIFA!)-10 Dataset.

Images from the CIFAR-10 Dataset using CNNs
MNIST is a beginner-friendly dataset in computer vision. It’s easy to score 90%+ on validation by using a CNN model. But what if you are beyond beginner and need something challenging to put your concepts to use?

That’s where the CIFAR-10 dataset comes into the picture!

CIFAR-10 CNN

Here’s how the developers behind CIFAR (Canadian Institute For Advanced Research) describe the dataset:

The CIFAR-10 dataset consists of 60,000 32 x 32 colour images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images.

The important points that distinguish this dataset from MNIST are:

Images are colored in CIFAR-10 as compared to the black and white texture of MNIST
Each image is 32 x 32 pixel
50,000 training images and 10,000 testing images
Now, these images are taken in varying lighting conditions and at different angles, and since these are colored images, you will see that there are many variations in the color itself of similar objects (for example, the color of ocean water). If you use the simple CNN architecture that we saw in the MNIST example above, you will get a low validation accuracy of around 60%.

That’s a key reason why I recommend CIFAR-10 as a good dataset to practice your hyperparameter tuning skills for CNNs. The good thing is that just like MNIST, CIFAR-10 is also easily available in Keras.

You can simply load the dataset using the following code:

from keras.datasets import cifar10
# loading the dataset 
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
Here’s how you can build a decent (around 78-80% on validation) CNN model for CIFAR-10. Notice how the shape values have been updated from (28, 28, 1) to (32, 32, 3) according to the size of the images:

# keras imports for the dataset and building our neural network
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.utils import np_utils

# loading the dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# # building the input vector from the 32x32 pixels
X_train = X_train.reshape(X_train.shape[0], 32, 32, 3)
X_test = X_test.reshape(X_test.shape[0], 32, 32, 3)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalizing the data to help with the training
X_train /= 255
X_test /= 255

# one-hot encoding using keras' numpy-related utilities
n_classes = 10
print("Shape before one-hot encoding: ", y_train.shape)
Y_train = np_utils.to_categorical(y_train, n_classes)
Y_test = np_utils.to_categorical(y_test, n_classes)
print("Shape after one-hot encoding: ", Y_train.shape)

# building a linear stack of layers with the sequential model
model = Sequential()

# convolutional layer
model.add(Conv2D(50, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', input_shape=(32, 32, 3)))

# convolutional layer
model.add(Conv2D(75, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(125, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# flatten output of conv
model.add(Flatten())

# hidden layer
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(250, activation='relu'))
model.add(Dropout(0.3))
# output layer
model.add(Dense(10, activation='softmax'))

# compiling the sequential model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# training the model for 10 epochs
model.fit(X_train, Y_train, batch_size=128, epochs=10, validation_data=(X_test, Y_test))
Increased the number of Conv2D layers to build a deeper model
Increased number of filters to learn more features
Added Dropout for regularization
Added more Dense layers
Training and validation accuracy across epochs:



You can easily eclipse this performance by tuning the above model. Once you have mastered CIFAR-10, there’s also CIFAR-100 available in Keras that you can use for further practice. Since it has 100 classes, it won’t be an easy task to achieve!
