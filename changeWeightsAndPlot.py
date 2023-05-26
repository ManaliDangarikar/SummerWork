import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the data
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
num_classes = 10
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# Load the saved model
loaded_model = tf.keras.models.load_model('my_model_may26.h5')

# Get the weights of the loaded model
model_weights = loaded_model.get_weights()

## Access the weights of each layer by index
#for i, layer_weights in enumerate(model_weights):
#    print("Layer", i)
#    print(layer_weights.shape)
#    print(layer_weights)  # Print the weights if desired
#    print()

# Create a copy of the model weights
modified_model_weights = [np.copy(w) for w in model_weights]

# Define the percentage of weights to randomly set to zero
# percent_to_zero = 10  # Modify this value to the desired percentage

# Define the percentages of weights to randomly set to zero
percentages = range(0, 101, 10)  # Vary from 0 to 100 with a step size of 10

# Lists to store the percentages and corresponding accuracies
percentage_list = []
accuracy_list = []

# Iterate over the percentages and evaluate the modified models
for percent_to_zero in percentages:
    # re-assign the model weights to original model
    modified_model_weights = [np.copy(w) for w in model_weights]
    # Iterate over the model's weights and randomly set a portion to zero
    for i, layer_weights in enumerate(modified_model_weights):
        # Check if the layer has weights (i.e., it's not a non-trainable layer like MaxPooling2D)
        if len(layer_weights) > 0:
            # Compute the number of weights to set to zero based on the percentage
            num_weights = layer_weights.size
            num_weights_to_zero = int(num_weights * percent_to_zero / 100)

            # Generate a random mask to select which weights to zero
            mask = np.random.choice(num_weights, num_weights_to_zero, replace=False)

            # Set the selected weights to zero
            layer_weights.reshape(-1)[mask] = 0

    # Set the modified weights to the model
    loaded_model.set_weights(modified_model_weights)

    # Evaluate the model
    score = loaded_model.evaluate(x_test, y_test, verbose=0)
    # print('Test loss:', score[0])
    print('Percentage of weights set to 0:', percent_to_zero, 'Test accuracy:', score[1])
    
    # Append the percentage and accuracy to the lists
    percentage_list.append(percent_to_zero)
    accuracy_list.append(score[1])

# Plot the graph of percentage vs accuracy
plt.plot(percentage_list, accuracy_list, marker='o')
plt.xlabel('Percentage of Weights Modified')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Percentage of Weights Modified')
plt.grid(True)
plt.show()