import numpy as np
import matplotlib.pyplot as plt
import os, cv2, urllib, urllib.request, zipfile
from Components.mapping import Mapping


# Dense layer
class Layer_Dense:

    # Layer initialization
    def __init__(self, inputs, neurons, weight_regularizer_l1=0, weight_regularizer_l2=0, bias_regularizer_l1=0,
                 bias_regularizer_l2=0):

        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(inputs, neurons)
        self.biases = np.zeros((1, neurons))
        # Set regularization strength
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    # Forward pass
    def forward(self, inputs, training):

        # Remember input values
        self.inputs = inputs
        # Calculate output values from input ones, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases

    # Backward pass
    def backward(self, dvalues):

        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # Gradients on regularization
        # L1 on weights
        if self.weight_regularizer_l1 > 0:
            dL1 = self.weights.copy()
            dL1[dL1 >= 0] = 1
            dL1[dL1 < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        # L2 on weights
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights
        # L1 on biases
        if self.bias_regularizer_l1 > 0:
            dL1 = self.biases.copy()
            dL1[dL1 >= 0] = 1
            dL1[dL1 < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        # L2 on biases
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases

        # Gradient on values
        self.dvalues = np.dot(dvalues, self.weights.T)

    def predictions(self, outputs):

        a = outputs[:, 0, :]
        b = outputs[:, 1, :]
        distances = np.linalg.norm(a - b, axis=1)

        return distances < 0.5


# Dropout
class Layer_Dropout:

    # Init
    def __init__(self, rate):

        # Store rate, we invert it as for example for dropout of 0.1 we need success rate of 0.9
        self.rate = 1 - rate

    # Forward pass
    def forward(self, values, training):

        # Save input values
        self.inputs = values

        # If not in the training mode - return values
        if not training:
            self.output = values.copy()
            return

        # Generate and save scaled mask
        self.binary_mask = np.random.binomial(1, self.rate, size=values.shape) / self.rate
        # Apply mask to output values
        self.output = values * self.binary_mask

    # Backward pass
    def backward(self, dvalues):

        # Gradient on values
        self.dvalues = dvalues * self.binary_mask


# Input "layer"
class Layer_Input:

    # Forward pass
    def forward(self, inputs, training):

        self.output = inputs


# ReLU activation
class Activation_ReLU:

    # Forward pass
    def forward(self, inputs, training):

        # Remember input values
        self.inputs = inputs
        # Calculate output values from inputs
        self.output = np.maximum(0, inputs)

    # Backward pass
    def backward(self, dvalues):

        # Since we need to modify original variable,
        # let's make a copy of values first
        self.dvalues = dvalues.copy()

        # Zero gradient where input values were negative
        self.dvalues[self.inputs <= 0] = 0

    # Calculate predictions for outputs
    def predictions(self, outputs):

        return outputs


# Softmax activation
class Activation_Softmax:

    # Forward pass
    def forward(self, inputs, training):

        # Remember input values
        self.inputs = inputs

        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1,
                                            keepdims=True))
        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1,
                                            keepdims=True)

        self.output = probabilities

    # Backward pass
    def backward(self, dvalues):

        self.dvalues = dvalues.copy()

    # Calculate predictions for outputs
    def predictions(self, outputs):

        return np.argmax(outputs, axis=1)


# Sigmoid activation
class Activation_Sigmoid:

    # Forward pass
    def forward(self, inputs, training):

        # Save input and calculate/save output of sigmoid function
        self.input = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    # Backward pass
    def backward(self, dvalues):

        # Derivative - calculates from output of sigmoid function
        self.dvalues = dvalues * (1 - self.output) * self.output

    # Calculate predictions for outputs
    def predictions(self, outputs):

        return (outputs > 0.5) * 1


# Linear activation
class Activation_Linear:

    # Forward pass
    def forward(self, inputs, training):

        # Just remember values
        self.input = inputs
        self.output = inputs

    # Backward pass
    def backward(self, dvalues):

        # 1 is derivative, 1 * dvalued = dvalues - chain rule
        self.dvalues = dvalues.copy()

    # Calculate predictions for outputs
    def predictions(self, outputs):

        return outputs


# SGD optimizer
class Optimizer_SGD:

    # Initialize optimizer - set settings,
    # learning rate of 1. is default for this optimizer
    def __init__(self, learning_rate=1., decay=0., momentum=0.):

        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    # Call once before any parameter updates
    def pre_update_params(self):

        if self.decay:
            self.current_learning_rate = self.current_learning_rate * \
                                         (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):

        # If we use momentum
        if self.momentum:

            # If layer does not contain momentum arrays, create ones
            # filled with zeros
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                # If there is no momentum array for weights
                # The array doesnâ€TMt exist for biases yet either.
                layer.bias_momentums = np.zeros_like(layer.biases)

            # Build weight updates with momentum - take previous
            # updates multiplied by retain factor and update with
            # current gradients
            weight_updates = (
                    (self.momentum * layer.weight_momentums) -
                    (self.current_learning_rate * layer.dweights)
            )
            layer.weight_momentums = weight_updates

            # Build bias updates
            bias_updates = (
                    (self.momentum * layer.bias_momentums) -
                    (self.current_learning_rate * layer.dbiases)
            )
            layer.bias_momentums = bias_updates

        # Vanilla SGD updates (as before momentum update)
        else:
            weight_updates = (-self.current_learning_rate *
                              layer.dweights)
            bias_updates = (-self.current_learning_rate *
                            layer.dbiases)

        # Update weights and biases using either
        # vanilla or momentum updates
        layer.weights += weight_updates
        layer.biases += bias_updates

    # Call once after any parameter updates
    def post_update_params(self):

        self.iterations += 1


# Adagrad optimizer
class Optimizer_Adagrad:

    # Initialize optimizer - set settings
    def __init__(self, learning_rate=1., decay=0., epsilon=1e-7):

        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    # Call once before any parameter updates
    def pre_update_params(self):

        if self.decay:
            self.current_learning_rate = self.current_learning_rate * \
                                         (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):

        # If layer does not contain cache arrays,
        # create ones filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update cache with squared current gradients
        layer.weight_cache += layer.dweights ** 2
        layer.bias_cache += layer.dbiases ** 2

        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += -self.current_learning_rate * \
                         layer.dweights / \
                         (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * \
                        layer.dbiases / \
                        (np.sqrt(layer.bias_cache) + self.epsilon)

    # Call once after any parameter updates
    def post_update_params(self):

        self.iterations += 1


# RMSprop optimizer
class Optimizer_RMSprop:

    # Initialize optimizer - set settings
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, rho=0.9):

        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    # Call once before any parameter updates
    def pre_update_params(self):

        if self.decay:
            self.current_learning_rate = self.current_learning_rate * \
                                         (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):

        # If layer does not contain cache arrays,
        # create ones filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update cache with squared current gradients
        layer.weight_cache = self.rho * layer.weight_cache + \
                             (1 - self.rho) * layer.dweights ** 2
        layer.bias_cache = self.rho * layer.bias_cache + \
                           (1 - self.rho) * layer.dbiases ** 2

        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += -self.current_learning_rate * \
                         layer.dweights / \
                         (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * \
                        layer.dbiases / \
                        (np.sqrt(layer.bias_cache) + self.epsilon)

    # Call once after any parameter updates
    def post_update_params(self):

        self.iterations += 1


# Adam optimizer
class Optimizer_Adam:

    # Initialize optimizer - set settings
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):

        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    # Call once before any parameter updates
    def pre_update_params(self):

        if self.decay:
            self.current_learning_rate = self.current_learning_rate * \
                                         (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):

        # If layer does not contain cache arrays,
        # create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update momentum  with current gradients
        layer.weight_momentums = self.beta_1 * \
                                 layer.weight_momentums + \
                                 (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * \
                               layer.bias_momentums + \
                               (1 - self.beta_1) * layer.dbiases
        # Get corrected momentum
        # self.iteration is 0 at first pass
        # and we need to start with 1 here
        weight_momentums_corrected = layer.weight_momentums / \
                                     (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / \
                                   (1 - self.beta_1 ** (self.iterations + 1))
        # Update cache with squared current gradients
        layer.weight_cache = self.beta_2 * layer.weight_cache + \
                             (1 - self.beta_2) * layer.dweights ** 2
        layer.bias_cache = self.beta_2 * layer.bias_cache + \
                           (1 - self.beta_2) * layer.dbiases ** 2
        # Get corrected cache
        weight_cache_corrected = layer.weight_cache / \
                                 (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / \
                               (1 - self.beta_2 ** (self.iterations + 1))

        # Vanilla SGD parameter update + normalization with square rooted cache
        layer.weights += -self.current_learning_rate * \
                         weight_momentums_corrected / \
                         (np.sqrt(weight_cache_corrected) +
                          self.epsilon)
        layer.biases += -self.current_learning_rate * \
                        bias_momentums_corrected / \
                        (np.sqrt(bias_cache_corrected) +
                         self.epsilon)

    # Call once after any parameter updates
    def post_update_params(self):

        self.iterations += 1


# Common loss class
class Loss:

    # Regularization loss calculation
    def regularization_loss(self):

        # 0 by default
        regularization_loss = 0

        # Calculate regularization loss - iterate all trainable layers
        for layer in self.trainable_layers:

            # L1 regularization - weights
            if layer.weight_regularizer_l1 > 0:  # only calculate when factor greater than 0
                regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))

            # L2 regularization - weights
            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)

            # L1 regularization - biases
            if layer.bias_regularizer_l1 > 0:  # only calculate when factor greater than 0
                regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))

            # L2 regularization - biases
            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)

        return regularization_loss

    # Set/remember trainable layers
    def remember_trainable_layers(self, trainable_layers):

        self.trainable_layers = trainable_layers

    # Calculates the data and regularization losses
    # given model output and ground truth values
    def calculate(self, output, y, *, include_regularization=False):

        # Calculate sample losses
        sample_losses = self.forward(output, y)

        # Calculate mean loss
        data_loss = np.mean(sample_losses)

        # Add accumulated sum of losses and sample count
        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)

        # If just data loss - return it
        if not include_regularization:
            return data_loss

        # Return the data and regularization losses
        return data_loss, self.regularization_loss()

    # Calculates accumulated loss
    def calculate_accumulated(self, *, include_regularization=False):

        # Calculate mean loss
        data_loss = self.accumulated_sum / self.accumulated_count

        # If just data loss - return it
        if not include_regularization:
            return data_loss

        # Return the data and regularization losses
        return data_loss, self.regularization_loss()

    # Reset variables for accumulated loss
    def new_pass(self):

        self.accumulated_sum = 0
        self.accumulated_count = 0


class Loss_Constrastive(Loss):

    def __init__(self, margin=1.0):

        self.margin = margin

    def forward(self, y_pred, y_true):

        # Split into anchor and paired embeddings
        a, b = y_pred[:, 0, :], y_pred[:, 1, :]

        # Compute Euclidean distances
        distances = np.linalg.norm(a - b, axis=1)

        # Compute losses
        positive_loss = y_true * (distances ** 2)
        negative_loss = (1 - y_true) * (np.maximum(0, self.margin - distances) ** 2)
        losses = positive_loss + negative_loss

        return losses

    def backward(self, dvalues, y_true):

        # Split out pairs
        a = dvalues[:, 0, :]
        b = dvalues[:, 1, :]
        # Calculate Euclidean distances between pairs
        diff = a - b
        distances = np.linalg.norm(diff, axis=1, keepdims=True)  # shape (batch_size, 1)
        # Prevent division by zero
        distances = np.maximum(distances, 1e-7)

        # Broadcast y_true to shape (batch_size, 1)
        y_true = y_true.reshape(-1, 1)

        # Compute masks
        pos_mask = y_true
        neg_mask = 1 - y_true
        margin_term = np.maximum(0, self.margin - distances)

        # Loss = distance ** 2, therefore gradient = 2 * (a - b)
        grad_pos = 2 * pos_mask * diff
        # Derivative for negative gradient
        grad_neg = -2 * neg_mask * margin_term * diff / distances
        # Total gradients
        grad = grad_pos + grad_neg

        # Save gradients for both parts
        self.dvalues = np.zeros_like(dvalues)
        self.dvalues[:, 0, :] = grad  # ∂L/∂a
        self.dvalues[:, 1, :] = -grad  # ∂L/∂b (opposite direction)


# Cross-entropy loss
class Loss_CategoricalCrossentropy(Loss):

    # Forward pass
    def forward(self, y_pred, y_true):

        # Number of samples in a batch
        samples = y_pred.shape[0]

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Probabilities for target values -
        # only if categorical labels
        if len(y_true.shape) == 1:
            y_pred_clipped = y_pred_clipped[range(samples), y_true]

        # Losses
        negative_log_likelihoods = -np.log(y_pred_clipped)

        # Mask values - only for one-hot encoded labels
        if len(y_true.shape) == 2:
            negative_log_likelihoods *= y_true

        # Return losses
        return negative_log_likelihoods

    # Backward pass
    def backward(self, dvalues, y_true):

        samples = dvalues.shape[0]

        self.dvalues = dvalues.copy()  # Copy so we can safely modify
        self.dvalues[range(samples), y_true] -= 1
        self.dvalues = self.dvalues / samples

class Loss_SparseCategoricalCrossentropy(Loss):

    # Forward pass
    def forward(self, y_pred, y_true):

        # Number of samples in a batch
        samples = y_pred.shape[0]

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        correct_confidences = y_pred_clipped[range(samples), y_true]
        negative_log_likelihoods = -np.log(correct_confidences)

        return negative_log_likelihoods

    # Backward pass
    def backward(self, dvalues, y_true):

        samples = dvalues.shape[0]
        labels = dvalues.shape[1]

        self.dvalues = dvalues.copy()
        self.dvalues[range(samples), y_true] -= 1
        self.dvalues = self.dvalues / samples

# Binary cross-entropy loss
class Loss_BinaryCrossentropy(Loss):

    # Forward pass
    def forward(self, y_pred, y_true):
        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Calculate sample-wise loss
        sample_losses = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))

        # Return losses
        return sample_losses

    # Backward pass
    def backward(self, dvalues, y_true):

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)

        # Gradient on values
        self.dvalues = -(y_true / clipped_dvalues - (1 - y_true) / (1 - clipped_dvalues))


# Mean Squared Error loss
class Loss_MeanSquaredError(Loss):  # L2 loss

    # Forward pass
    def forward(self, y_pred, y_true):

        # Calculate loss
        data_loss = 2 * np.mean((y_true - y_pred) ** 2, axis=-1)

        # Return losses
        return data_loss

    # Backward pass
    def backward(self, dvalues, y_true):

        # Gradient on values
        self.dvalues = -(y_true - dvalues)


# Mean Absolute Error loss
class Loss_MeanAbsoluteError(Loss):  # L1 loss

    def forward(self, y_pred, y_true):

        # Calculate loss
        data_loss = np.mean(np.abs(y_true - y_pred), axis=-1)

        # Return losses
        return data_loss

    # Backward pass
    def backward(self, dvalues, y_true):

        # Gradient on values
        self.dvalues = -np.sign(y_true - dvalues)


# Common accurcy class
class Accuracy:

    # Calculates an accuracy
    # given predictions and ground truth values
    def calculate(self, predictions, y):

        # Get comparison results
        comparisons = self.compare(predictions, y)

        # Calculate an accuracy
        accuracy = np.mean(comparisons)

        # Add accumulated sum of matching values and sample count
        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)

        # Return accuracy
        return accuracy

    # Calculates accumulated accuracy
    def calculate_accumulated(self):

        # Calculate an accuracy
        accuracy = self.accumulated_sum / self.accumulated_count

        # Return the data and regularization losses
        return accuracy

    # Reset variables for accumulated accuracy
    def new_pass(self):

        self.accumulated_sum = 0
        self.accumulated_count = 0


class Accuracy_Constrastive(Accuracy):

    def __init__(self, threshold=0.5):

        self.threshold = threshold

    def init(self, y):

        pass

    def compare(self, predictions, y):

        return predictions == y

# Accuracy calculation for classification model
class Accuracy_Categorical(Accuracy):

    # No initialization is needed
    def init(self, y):

        pass

    # Compares predictions to the ground truth values
    def compare(self, predictions, y):

        return predictions == y


# Accuracy calculation for regression model
class Accuracy_Regression(Accuracy):

    def __init__(self):

        # Create precision property
        self.precision = None

    # Calculates precision value based on passed in ground truth values
    def init(self, y, reinit=False):

        if self.precision is None or reinit:
            self.precision = np.std(y) / 500

    # Compares predictions to the ground truth values
    def compare(self, predictions, y):

        return np.absolute(predictions - y) < self.precision


# Model class
class Model:

    def __init__(self):

        # Create a list of network objects
        self.layers = []

    # Add objects to the model
    def add(self, layer):

        self.layers.append(layer)

    # Set loss, optimizer and accuracy
    def set(self, *, loss, optimizer, accuracy):

        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

    # Finalize the model
    def finalize(self):

        # Create and set the input layer
        self.input_layer = Layer_Input()

        # Count all the objects
        layer_count = len(self.layers)

        # Initialize a list containing trainable layers:
        self.trainable_layers = []

        # Iterate the objects
        for i in range(layer_count):

            # If it's the first one, the previous called object will be the input layer
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i + 1]

            # All layers except for the first and the last
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.layers[i + 1]

            # The last layer - the next object is the loss
            # Also let's save aside the reference to the last object
            # whose output is the model's output
            else:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            # If layer contains an attribute called "weights", it's a trainable layer
            # add it to the list of trainable layers
            # We don;t need to check for biases - checking for weights is enough
            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])

            # Update loss object with trainable layers
            self.loss.remember_trainable_layers(
                self.trainable_layers
            )

    # Train the model
    def train(self, X, y, *, epochs=1, batch_size=None, print_every=1, validation_data=None):

        # Initialize accuracy object
        self.accuracy.init(y)

        # Default value if batch size is not set
        train_steps = 1

        # If there is validation data passed,
        # set default number of steps for validation as well
        if validation_data is not None:
            validation_steps = 1

            # For better readability
            X_val, y_val = validation_data

        # Calculate number of steps
        if batch_size is not None:
            train_steps = len(X) // batch_size
            # Dividing rounds down. If there are some remaining data,
            # but not a full batch, this won't include it
            # Add `1` to include this not full batch
            if train_steps * batch_size < len(X):
                train_steps += 1

            if validation_data is not None:
                validation_steps = len(X_val) // batch_size
                # Dividing rounds down. If there are some remaining data, but nor full batch, this won't include it
                # Add `1` to include this not full batch
                if validation_steps * batch_size < len(X_val):
                    validation_steps += 1

        # Main training loop
        for epoch in range(1, epochs + 1):

            # Print epoch number
            # print(f'epoch: {epoch}')

            # Reset accumulated values in loss and accuracy objects
            self.loss.new_pass()
            self.accuracy.new_pass()

            # Iterate over steps
            for step in range(train_steps):

                # If batch size is not set - train using one step and full dataset
                if batch_size is None:
                    batch_X = X
                    batch_y = y

                # Otherwise slice a batch
                else:
                    batch_X = X[step * batch_size:(step + 1) * batch_size]
                    batch_y = y[step * batch_size:(step + 1) * batch_size]

                # Perform the forward pass
                output = self.forward(batch_X, training=True)

                # Calculate loss
                data_loss, regularization_loss = self.loss.calculate(output, batch_y, include_regularization=True)
                loss = data_loss + regularization_loss

                # Get predictions and calculate an accuracy
                predictions = self.output_layer_activation.predictions(output)
                accuracy = self.accuracy.calculate(predictions, batch_y)

                # Perform backward pass
                self.backward(output, batch_y)

                # Optimize (update parameters)
                self.optimizer.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params()

                # Print a summary
                # if not step % print_every or step == train_steps - 1:
                #     print(
                #         f'step: {step}, acc: {accuracy:.3f}, loss: {loss:.3f} (data_loss: {data_loss:.3f}, reg_loss: {regularization_loss:.3f}), lr: {self.optimizer.current_learning_rate}')

            # Get and print epoch loss and accuracy
            epoch_data_loss, epoch_regularization_loss = self.loss.calculate_accumulated(include_regularization=True)
            epoch_loss = epoch_data_loss + epoch_regularization_loss
            epoch_accuracy = self.accuracy.calculate_accumulated()

            # print(
            #     f'training, acc: {epoch_accuracy:.3f}, loss: {epoch_loss:.3f} (data_loss: {epoch_data_loss:.3f}, reg_loss: {epoch_regularization_loss:.3f}), lr: {self.optimizer.current_learning_rate}')

            # If there is the validation data
            if validation_data is not None:

                # Reset accumulated values in loss and accuracy objects
                self.loss.new_pass()
                self.accuracy.new_pass()

                # Iterate over steps
                for step in range(validation_steps):

                    # If batch size is not set - train using one step and full dataset
                    if batch_size is None:
                        batch_X = X_val
                        batch_y = y_val

                    # Otherwise slice a batch
                    else:
                        batch_X = X_val[step * batch_size:(step + 1) * batch_size]
                        batch_y = y_val[step * batch_size:(step + 1) * batch_size]

                    # Perform the forward pass
                    output = self.forward(batch_X, training=False)

                    # Calculate the loss
                    self.loss.calculate(output, batch_y)

                    # Get predictions and calculate an accuracy
                    predictions = self.output_layer_activation.predictions(output)
                    self.accuracy.calculate(predictions, batch_y)

                # Get and print validation loss and accuracy
                validation_loss = self.loss.calculate_accumulated()
                validation_accuracy = self.accuracy.calculate_accumulated()

                # Print a summary
                # print(f'validation, acc: {validation_accuracy:.3f}, loss: {validation_loss:.3f}')

    # Performs forward pass
    def forward(self, X, training):

        if X.ndim == 3:
            # Contrastive input: shape (batch_size, 2, input_dim)
            input_a = X[:, 0, :]
            input_b = X[:, 1, :]

            # === FORWARD A ===
            self.input_layer.forward(input_a, training)
            for layer in self.layers:
                layer.forward(layer.prev.output, training)
                layer.inputs_a = layer.inputs.copy()

            output_a = self.layers[-1].output

            # === FORWARD B ===
            self.input_layer.forward(input_b, training)
            for layer in self.layers:
                layer.forward(layer.prev.output, training)
                layer.inputs_b = layer.inputs.copy()

            output_b = self.layers[-1].output

            return np.stack([output_a, output_b], axis=1)

        else:
            self.input_layer.forward(X, training)

            # Call forward method of every object in a chain
            # Pass output of the previous object as a parameter
            for layer in self.layers:
                layer.forward(layer.prev.output, training)

            # "layer" is now the last object from the list,
            # return it's output
            return layer.output

    # Performs backward pass
    def backward(self, output, y):

        # Backward pass through loss function to get initial dvalues
        self.loss.backward(output, y)

        # Check dimensionality
        if self.loss.dvalues.ndim == 3:
            # Constrastive backpropogation (batch_size, 2, embedding_dim). Split out pairs
            dvalues_a = self.loss.dvalues[:, 0, :]
            dvalues_b = self.loss.dvalues[:, 1, :]
            # Backward pass for a
            for layer in self.layers:
                layer.inputs = layer.inputs_a
            # Backward from last to first
            reversed_layers = list(reversed(self.layers))
            reversed_layers[0].backward(dvalues_a)  # pass in explicitly to last layer
            for i in range(1, len(reversed_layers)):
                reversed_layers[i].backward(reversed_layers[i - 1].dvalues)

            # Backward pass for b
            for layer in self.layers:
                layer.inputs = layer.inputs_b
            reversed_layers[0].backward(dvalues_b)
            for i in range(1, len(reversed_layers)):
                reversed_layers[i].backward(reversed_layers[i - 1].dvalues)
        else:
            for layer in reversed(self.layers):

                layer.backward(layer.next.dvalues)


def generate_contrastive_pairs(X, y, num_pairs=100):

    pairs = []
    labels = []

    unique_classes = np.unique(y)

    for _ in range(num_pairs):

        if np.random.rand() < 0.5:
            # Positive pair (same class)
            label = 1
            cls = np.random.choice(unique_classes)
            indices = np.where(y == cls)[0]
            a, b = np.random.choice(indices, size=2, replace=False)
        else:
            # Negative pair (different class)
            label = 0
            cls_a, cls_b = np.random.choice(unique_classes, size=2, replace=False)
            a = np.random.choice(np.where(y == cls_a)[0])
            b = np.random.choice(np.where(y == cls_b)[0])

        pairs.append((X[a], X[b]))
        labels.append(label)

    return np.array(pairs), np.array(labels)

if __name__ == "__main__":

    X = np.array([[0.9] * 3 for _ in range(20)] + [[0.1] * 3 for _ in range(20)]+ [[0.5] * 3 for _ in range(20)])
    y = np.array([1] * 20 + [0] * 20 + [0.5] * 20)
    X_test = np.array([[0.9, 0.9, 0.9], [0.1, 0.1, 0.1], [0.5, 0.5, 0.5]])
    y_test = np.array([1, 0, 0.5])

    pairs, pair_labels = generate_contrastive_pairs(X, y)

    network = Model()

    network.add(Layer_Dense(3, 512, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4))
    network.add(Activation_ReLU())
    network.add(Layer_Dense(512, 512))
    network.add(Activation_ReLU())
    network.add(Layer_Dropout(rate=0.1))
    network.add(Layer_Dense(512, 3))

    # Set loss, optimizer and accuracy objects
    network.set(
        loss=Loss_Constrastive(),
        optimizer=Optimizer_Adam(decay=1e-7),
        accuracy=Accuracy_Constrastive()
    )

    network.finalize()

    network.train(X=pairs, y=pair_labels, epochs=1, batch_size=128)
    print("Before training:")

    e1 = network.forward(np.array([0.9, 0.9, 0.9]), training=None)
    e2 = network.forward(np.array([0.1, 0.1, 0.1]), training=None)
    e3 = network.forward(np.array([0.5, 0.5, 0.5]), training=None)

    print(f"Input array: {np.array([0.9, 0.9, 0.9])}, embedding: {e1}")
    print("Distance between 0.9 and 0.1:", np.linalg.norm(e1 - e2))
    print(f"Input array: {np.array([0.1, 0.1, 0.1])}, embedding: {e2}")
    print("Distance between 0.9 and 0.5:", np.linalg.norm(e1 - e3))
    print(f"Input array: {np.array([0.5, 0.5, 0.5])}, embedding: {e3}")
    print("Distance between 0.1 and 0.5:", np.linalg.norm(e2 - e3))

    """HEATMAP CODE
    """
    embeddings = []
    embeddings_test = []
    for x in X:

        out = network.forward(x, training=None)
        embeddings.append(out[0])

    for x in X_test:

        out = network.forward(x, training=None)
        embeddings_test.append(out[0])

    embeddings = np.array(embeddings)  # Shape (60, 3)
    embeddings_test = np.array(embeddings_test)  # Shape (60, 3)

    # Create instance of embedding heatmap
    map = Mapping()

    # Update new points into map
    for i in range(len(X)):

        map.update(embeddings[i], y[i])

    for i in range(len(X_test)):

        score = map.score(embeddings_test[i])
        print(f"Input: {X_test[i]}, Label: {y_test[i]}, Embedding: {embeddings_test[i]}, Predicted Label: {score}")


    network.train(X=pairs, y=pair_labels, epochs=1000, batch_size=128)

    print("After training:")
    e1 = network.forward(np.array([0.9, 0.9, 0.9]), training=None)
    e2 = network.forward(np.array([0.1, 0.1, 0.1]), training=None)
    e3 = network.forward(np.array([0.5, 0.5, 0.5]), training=None)

    print(f"Input array: {np.array([0.9, 0.9, 0.9])}, embedding: {e1}")
    print("Distance between 0.9 and 0.1:", np.linalg.norm(e1 - e2))
    print(f"Input array: {np.array([0.1, 0.1, 0.1])}, embedding: {e2}")
    print("Distance between 0.9 and 0.5:", np.linalg.norm(e1 - e3))
    print(f"Input array: {np.array([0.5, 0.5, 0.5])}, embedding: {e3}")
    print("Distance between 0.1 and 0.5:", np.linalg.norm(e2 - e3))

    """HEATMAP CODE
        """
    embeddings = []
    embeddings_test = []
    for x in X:
        out = network.forward(x, training=None)
        embeddings.append(out[0])

    for x in X_test:
        out = network.forward(x, training=None)
        embeddings_test.append(out[0])

    embeddings = np.array(embeddings)  # Shape (60, 3)
    embeddings_test = np.array(embeddings_test)  # Shape (60, 3)

    # Create instance of embedding heatmap
    map = Mapping()

    # Update new points into map
    for i in range(len(X)):
        map.update(embeddings[i], y[i])

    for i in range(len(X_test)):
        score = map.score(embeddings_test[i])
        print(f"Input: {X_test[i]}, Label: {y_test[i]}, Embedding: {embeddings_test[i]}, Predicted Label: {score}")