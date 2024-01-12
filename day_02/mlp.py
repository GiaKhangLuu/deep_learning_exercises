import numpy as np
import matplotlib.pyplot as plt
import os

class ANN:

    def __init__(self):
        pass

    def initialize_parameters(self, layer_dims):
        np.random.seed(1)
        parameters = {}
        L  = len(layer_dims)

        for l in range(1, L):
            # Initializing weights and bias at layer l with normal dist.
            weights_at_l = np.random.standard_normal((layer_dims[l], layer_dims[l - 1]))
            bias_at_l = np.random.standard_normal((layer_dims[l], 1))

            # Limiting weights and bias in range [0, 0.01]
            #parameters['W' + str(l)] = np.clip(weights_at_l, 0.0, 0.01)
            #parameters['b' + str(l)] = np.clip(bias_at_l, 0.0, 0.01)
            parameters['W' + str(l)] = weights_at_l
            parameters['b' + str(l)] = bias_at_l

            assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
            assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

        return parameters
    
    def linear_forward(self, A, W, b):
        #Z = np.dot(W, A) + b
        Z = W.dot(A) + b

        assert (Z.shape == (W.shape[0], A.shape[1]))
        cache = (A, W, b)

        return Z, cache
    
    def sigmoid(self, Z):
        A = 1 / (1 + np.exp(-Z))
        cache = Z
        
        return A, cache

    def relu(self, Z):
        A = np.maximum(Z, 0)
        cache = Z

        return A, cache

    def linear_activation_forward(self, A, W, b, activation):
        Z, linear_cache = self.linear_forward(A, W, b)
        if activation == 'sigmoid':
            A, activation_cache = self.sigmoid(Z)
        elif activation == 'relu':
            A, activation_cache = self.relu(Z)
        else:
            print('Invalid activation function')
            return
        cache = (linear_cache, activation_cache)    

        return A, cache
    
    def model_forward(self, X, parameters):
        caches = []
        A = X
        L = len(parameters) // 2 

        for l in range(1, L):
            A_prev = A
            W = parameters['W' + str(l)]
            b = parameters['b' + str(l)]
            A, cache = self.linear_activation_forward(A_prev, W, b, 'relu')
            caches.append(cache)

        # Computing A in the last layer
        W = parameters['W' + str(l + 1)]
        b = parameters['b' + str(l + 1)]
        AL, cache = self.linear_activation_forward(A, W, b, 'sigmoid')
        caches.append(cache)

        return AL, caches
    
    def compute_cost(self, AL, Y):
        m = Y.shape[1]
        cost = -1 / m * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL)))
        #cost = -1 / m * np.sum(Y * np.log(AL) + ((1 - Y) * np.log(1 - AL)))
        cost = np.squeeze(cost)

        return cost
    
    def linear_backward(self, dZ, cache):
        A_pre, W, b = cache
        m = A_pre.shape[1]

        dW = 1/m * np.dot(dZ, A_pre.T)
        db = 1/m * np.sum(dZ, axis=1, keepdims=True)
        dA_pre = np.dot(W.T, dZ)

        return dA_pre, dW, db
    
    def relu_deravative(self, dA, cache):
        Z = cache
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0

        assert (dZ.shape == Z.shape) 

        return dZ

    def sigmoid_deravative(self, dA, cache):
        Z = cache
        s, _ = self.sigmoid(Z)
        dZ = dA * s * (1 - s)

        assert (dZ.shape == Z.shape)

        return dZ

    def linear_activation_backward(self, dA, cache, activation):
        linear_cache, activation_cache = cache

        if activation == 'relu':
            dZ = self.relu_deravative(dA, activation_cache)
        if activation == 'sigmoid':
            dZ = self.sigmoid_deravative(dA, activation_cache)
        
        dA_prev, dW, db = self.linear_backward(dZ, linear_cache)

        return dA_prev, dW, db
    
    def model_backward(self, AL, Y, caches):
        gradients = {}
        L = len(caches)  # Num of layers
        m = AL.shape[1]

        # 1. Compute dAL
        dAL = -1/m * (Y / AL - ((1 - Y) / (1 - AL)))

        # 2. Compute dW
        current_cache = caches[L - 1]
        dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(dAL, current_cache, 'sigmoid')
        gradients["dA" + str(L - 1)] = dA_prev_temp
        gradients["dW" + str(L)] = dW_temp
        gradients["db" + str(L)] = db_temp

        # Loop from l = L - 2 to l = 0
        for l in range(L - 1)[::-1]:
            # l-th layer: (RELU -> LINEAR) gradient
            # Input: grads["dA" + str(l + 1)], current_cache
            # Output: grads["dA" + str(l)], grads["dW" + str(l + 1)], grads["db" + str(l + 1)]
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(dA_prev_temp, current_cache, 'relu')
            gradients["dA" + str(l)] = dA_prev_temp
            gradients["dW" + str(l + 1)] = dW_temp
            gradients["db" + str(l + 1)] = db_temp

        return gradients
    
    def update_parameters(self, params, gradients, learning_rate):
        parameters = params.copy()
        L = len(parameters) // 2  # Num of layers

        for l in range(L):
            W_old, b_old = parameters["W" + str(l + 1)], parameters["b" + str(l + 1)]
            dW, db = gradients["dW" + str(l + 1)], gradients["db" + str(l + 1)]

            parameters["W" + str(l + 1)] = W_old - learning_rate * dW
            parameters["b" + str(l + 1)] = b_old - learning_rate * db
        
        return parameters
    
    def predict(self, X, y, parameters):
        m = X.shape[1]
        p = np.zeros((1, m))

        # Feed forward
        probas, _ = self.model_forward(X, parameters)
        p = probas.copy()

        if np.max(y) > 1:
            predicted_cls = np.argmax(p, axis=0)
        else:
            # Convert probas to 0/1 predictions
            i, j = np.where(p > 0.5)
            p[i, j] = 1
            i, j = np.where(p <= 0.5)
            p[i, j] = 0

        #print("Predictions: ", str(p))
        #print("Ground truth label: ", str(y))
        acc = np.sum((predicted_cls == y)/m)
        return acc
    
    def build_model(self, X, Y, layers_dims, learning_rate=0.1, num_iterations=1000, verbose=False, is_draw=False):
        np.random.seed(1)
        costs = []

        # Parameter initialization
        parameters = self.initialize_parameters(layers_dims)

        # Loop (gradient descent)
        for i in range(num_iterations):
            # Feed forward: [LINEAR -> RELU] * (L - 1) -> LINEAR -> SIGMOID
            AL, caches = self.model_forward(X, parameters)

            # Compute cost
            cost = self.compute_cost(AL, Y)

            # Backward propagation
            grads = self.model_backward(AL, Y, caches)

            # Update parameters
            parameters = self.update_parameters(parameters, grads, learning_rate)

            # Print cost after each 100 epochs
            if verbose and i % 500 == 0 or i == num_iterations - 1:
                print(f"Cost after iteration {i}: {np.squeeze(cost)}")

                # Draw boundaries at epoch i
                if is_draw:
                    save_dir = './asset'
                    fn = 'Epoch: {}.png'.format(i)
                    self.draw_boundaries(parameters, saved_path=os.path.join(save_dir, fn), 
                                         is_plot_data_points=True, fig_title=fn.split('.')[0])

            if i % 100 == 0 or i == num_iterations:
                costs.append(cost)
        
        return parameters, costs
    
    def draw_boundaries(self, parameters, saved_path, is_plot_data_points=False, fig_title=None):
        xm = np.arange(-1.5, 1.5, 0.025)
        xlen = len(xm)
        ym = np.arange(-1.5, 1.5, 0.025)
        ylen = len(ym)
        xx, yy = np.meshgrid(xm, ym)

        xx1 = xx.ravel().reshape(1, xx.size)
        yy1 = yy.ravel().reshape(1, yy.size)

        X = np.vstack((xx1, yy1))
        pred, _ = self.model_forward(X, parameters)

        # predicted class 
        Z = np.argmax(pred, axis=0)

        Z = Z.reshape(xx.shape)

        CS = plt.contourf(xx, yy, Z, 200, cmap='jet', alpha = .3)

        if is_plot_data_points:
            X, y, data_per_class = self.generate_data_points()
            self.plot_data_points(X, data_per_class)

        plt.xlim([-1.5, 1.5])
        plt.ylim([-1.5, 1.5])
        cur_axes = plt.gca()
        cur_axes.axes.get_xaxis().set_ticks([])
        cur_axes.axes.get_yaxis().set_ticks([])

        plt.xticks(())
        plt.yticks(())
        if fig_title:
            acc = self.predict(X, y, parameters)
            title = fig_title + " - Acc: {}%".format(str(round(acc * 100, 2)))
            plt.title(title, fontsize=10)

        plt.savefig(saved_path, bbox_inches='tight')

        plt.close()

    def generate_data_points(self):
        N = 100 # number of points per class
        d0 = 2 # dimensionality
        C = 3 # number of classes
        X = np.zeros((d0, N*C)) # data matrix (each row = single example)
        y = np.zeros(N*C, dtype='uint8') # class labels

        for j in range(C):
            np.random.seed(1)
            ix = range(N*j,N*(j+1))
            r = np.linspace(0.0,1,N) # radius
            t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
            X[:,ix] = np.c_[r*np.sin(t), r*np.cos(t)].T
            y[ix] = j

        return X, y, N

    def plot_data_points(self, data, data_per_class):
        plt.plot(data[0, :data_per_class], data[1, :data_per_class], 'bs', markersize = 7, markeredgecolor='black')
        plt.plot(data[0, data_per_class:2*data_per_class], data[1, data_per_class:2*data_per_class], 'g^', markersize = 7, markeredgecolor='black')
        plt.plot(data[0, 2*data_per_class:], data[1, 2*data_per_class:], 'ro', markersize = 7, markeredgecolor='black')


