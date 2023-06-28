import h5py
import cleverhans
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

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

model = Net()

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
## Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32)
print('Modify test images: Start!')

# Evaluate the model
score1 = model.evaluate(x_test, y_test)
print('Test loss:', score1[0]) 
print('Test accuracy:', score1[1])

# Evaluate on adversarial data
test_acc_clean = tf.metrics.SparseCategoricalAccuracy()
test_acc_fgsm = tf.metrics.SparseCategoricalAccuracy()

# progress_bar_test = tf.keras.utils.Progbar(10000)
# for j in x_test:
y_pred = model(x_test)
test_acc_clean(y_test, y_pred)

x_fgm = fast_gradient_method(model, x_test, 0.07, np.inf)
y_pred_fgm = model(x_fgm)
test_acc_fgsm(y_test, y_pred_fgm)

print('Modify test images: Complete!')

print(
        "test acc on clean examples (%): {:.3f}".format(test_acc_clean.result() * 100)
    )
print(
        "test acc on FGM adversarial examples (%): {:.3f}".format(
            test_acc_fgsm.result() * 100
        )
    )

model.save_weights('weights.h5')
subclassedModel_weights = model.get_weights()
for weight in subclassedModel_weights:
    print(weight.shape)
