import os
import tensorflow as tf
from tensorflow import keras

# Load the data

# Get current working directory
current_dir = os.getcwd()

# Append data/mnist.npz to the previous path to get the full path
data_path = os.path.join(current_dir, "assignment/data/mnist.npz")

# Discard test set
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path=data_path)
        
# Normalize pixel values
x_train = x_train / 255.0

data_shape = x_train.shape

print(f"There are {data_shape[0]} examples with shape ({data_shape[1]}, {data_shape[2]})")

# GRADED CLASS: myCallback


# Remember to inherit from the correct class
class myCallback(tf.keras.callbacks.Callback):
        # Define the correct function signature for on_epoch_end
        def on_epoch_end(self, epoch, logs={}):
            if logs.get('accuracy') is not None and logs.get('accuracy') > 0.99:
                print("\nReached 99% accuracy so cancelling training!") 
                
                # Stop training once the above condition is met
                self.model.stop_training = True


# GRADED FUNCTION: train_mnist
def train_mnist(x_train, y_train):
    
    # Instantiate the callback class
    callbacks = myCallback()
    
    # Define the model
    model = tf.keras.models.Sequential([ 
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        

    ]) 

    # Compile the model
    model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    
    # Fit the model for 10 epochs adding the callbacks
    # and save the training history
    history = model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])
    return history

hist = train_mnist(x_train, y_train)