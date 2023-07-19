# This file trains a 7-layer CNN over 10 epochs and evaluates it against clean, fgsm and pgdm test data set.
# It further modifies random weights to 0(step size of 2% per iteration, max zeroed weights 50 to reduce run time as accuracy remains 0.1 after 40%)
# and checks the accuracy for clean, fgsm and pgdm samples.
# epsilon values are x/ 255 where x is fibonacci sequence from 5 to < 255. Excluded 0 to 3 because it results in epsilon value of less than 0.01.
# since pgdm iterates with stepsize of 0.01, values of epsilon lower than 0.01 throw an error. 
# For each value of epsilon, we run the experiment 5 times to further calculate the mean test accuracy and standard deviation in "plotResultsfromExperiments.py"

import h5py
import cleverhans
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib as plt

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

print(epsilon_set)
for j in range(len(epsilon_set)):
    # Create lists to store the results
    fgsm_results = []
    pgd_results = []
    clean_results = []
    epsilon = epsilon_set[j]

    for i in range(5):
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

        # Get the weights of the loaded model
        model_weights = model.get_weights()

        # Create a copy of the model weights
        modified_model_weights = [np.copy(w) for w in model_weights]

        # Define the percentages of weights to randomly set to zero
        percentages = range(0, 52, 2)  # Vary from 0 to 50 with a step size of 2

        # Lists to store the percentages and corresponding accuracies
        percentage_list = []
        accuracy_list = []
        adversarial_accuracy_list = []
        set_percent_list = []
        actual_percent_list = []

        # Lists to store the accuracies for each evaluation
        fgsm_accuracies = []
        pgd_accuracies = []
        clean_accuracies = []

        # Iterate over the percentages and evaluate the modified models
        for percent_to_zero in percentages:
            # re-assign the model weights to original model
            modified_model_weights = [np.copy(w) for w in model_weights]
            total_zero_weights = 0
            total_weights = 0
            for i, layer_weights in enumerate(modified_model_weights):
                # Check if the layer has weights (i.e., it's not a non-trainable layer like MaxPooling2D)
                num_zero_weights_layer = 0
                if len(layer_weights) > 0:
                    # Compute the number of weights to set to zero based on the percentage
                    num_weights = layer_weights.size
                    num_weights_to_zero = int(num_weights * percent_to_zero / 100)

                    # Generate a random mask to select which weights to zero
                    mask = np.random.choice(num_weights, num_weights_to_zero, replace=False)
                    # check replace=false 
                    # replace: (optional) This parameter determines whether the sampling should be done with replacement or not. 
                    # If True, the same element can be selected multiple times. If False, each element can only be selected once.
                    # Set the selected weights to zero
                    layer_weights.reshape(-1)[mask] = 0
                    # In this case, the expression arr == 0 creates a boolean array where each element is True if the corresponding element in arr is equal to zero, and False otherwise.
                    # So, the resulting boolean array would be [True, False, False, True, False, False, True].
                    # Then, np.count_nonzero(arr == 0) counts the number of True values in the boolean array, which corresponds to the number of elements in arr that are equal to zero.
                    num_zero_weights_layer += np.count_nonzero(layer_weights == 0)
                    total_zero_weights += np.count_nonzero(layer_weights == 0)
                    total_weights += layer_weights.size

            total_zero_percentage = total_zero_weights/total_weights * 100.0
            actual_percent_list.append(total_zero_percentage)
            set_percent_list.append(percent_to_zero)

            print('Set zero percentage: ', percent_to_zero, '\tActual zero percentage: ', total_zero_percentage)

            # Set the modified weights to the model
            model.set_weights(modified_model_weights)

            # Evaluate clean test samples on model
            score = model.evaluate(x_test, y_test)
     
            # Evaluate fgsm test samples on model
            score1 = model.evaluate(x_fgm, y_test)

            # Evaluate pgd test samples on model
            score2 = model.evaluate(x_pgd, y_test)

            clean_accuracies.append(score[1])
            fgsm_accuracies.append(score1[1])
            pgd_accuracies.append(score2[1])


            print('Test accuracy clean:', score[1], '\tTest accuracy fgsm:', score1[1], '\tTest accuracy pgdm:', score2[1])

        # Append the accuracies for this percentage to the results list
        fgsm_results.append(fgsm_accuracies)
        pgd_results.append(pgd_accuracies)
        clean_results.append(clean_accuracies)

    # Convert the results to NumPy arrays for easier manipulation
    fgsm_results = np.array(fgsm_results)
    pgd_results = np.array(pgd_results)
    clean_results = np.array(clean_results)

    print(fgsm_results)
    print(pgd_results)
    print(clean_results)