# display clean, fgsm and pgdm test sample

import h5py
import cleverhans
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model

from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method

class Net(Model):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1))
        self.conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')
        self.pool1 = MaxPooling2D((2, 2), strides=(2, 2))

        self.conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')
        self.conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')
        self.pool2 = MaxPooling2D((2, 2), strides=(2, 2))

        self.conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')
        self.conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')
        self.conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')
        self.pool3 = MaxPooling2D((2, 2), strides=(2, 2))

        self.flatten = Flatten()
        self.dense1 = Dense(4096, activation="relu")
        self.dense2 = Dense(4096, activation="relu")
        self.dense3 = Dense(10, activation="softmax")

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.dense3(x)

# Prepare the data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

#fibonacci_seq = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233]
fibonacci_seq = [5, 8, 13, 21, 34, 55, 89, 144, 233]

epsilon_set = [num / 255 for num in fibonacci_seq]

#epsilon_set = [ 0/255, 1/255, 2/255, 3/255, 4/255, 5/255, 10/255, 15/255, 20/255, 25/255]
print(epsilon_set)

model = Net()

model.build(input_shape=(None, 28, 28, 1))
model.summary()

for j in range(len(epsilon_set)):
#    # Create lists to store the results
#    fgsm_results = []
#    pgd_results = []
#    clean_results = []
     epsilon = epsilon_set[j]

    for i in range(1):
        model = Net()

        # Compile the model
        model.compile(optimizer='sgd',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        # Train the model
        model.fit(x_train, y_train, epochs=10, batch_size=32)

        print('########################')
        print('Epsilon:', epsilon)

        # Evaluate clean test samples on model
        score = model.evaluate(x_test, y_test)

        # Create fgsm test samples
        x_fgm = fast_gradient_method(model, x_test, epsilon, np.inf)
   
        # Evaluate fgsm test samples on model
        score1 = model.evaluate(x_fgm, y_test)

        # Create pgd test samples
        x_pgd = projected_gradient_descent(model, x_test, epsilon, 0.01, 40, np.inf)

        # Evaluate pgd test samples on model
        score2 = model.evaluate(x_pgd, y_test)

        print('Run:', i, 'Test accuracy clean:', score[1], '\tTest accuracy fgsm:', score1[1], '\tTest accuracy pgdm:', score2[1])

        sample_index = 1

        # Select the specific FGSM sample for visualization
        x_fgm_sample = x_fgm[sample_index]
  
        # Select the specific PGD sample for visualization
        x_pgd_sample = x_pgd[sample_index]

        # Visualize the original image
        plt.figure()
        plt.imshow(x_test[sample_index].reshape(28, 28), cmap='gray')
        plt.title('Original Image (epsilon = {:.4f})'.format(epsilon))
        plt.savefig('original_image_{}.png'.format(epsilon))  # Save the figure as 'original_image.png'
        plt.show()

        # Visualize the FGSM sample
        plt.figure()
        plt.imshow(x_fgm_sample.numpy().reshape(28, 28), cmap='gray')
        plt.title('FGSM Attack (epsilon = {:.4f})'.format(epsilon))
        plt.savefig('fgsm_sample_{}.png'.format(epsilon))  # Save the figure as 'fgsm_sample.png'
        plt.show()

        # Visualize the PGD sample
        plt.figure()
        plt.imshow(x_pgd_sample.numpy().reshape(28, 28), cmap='gray')
        plt.title('PGD Attack (epsilon = {:.4f})'.format(epsilon))
        plt.savefig('pgd_sample_{}.png'.format(epsilon))  # Save the figure as 'pgd_sample.png'
        plt.show()
