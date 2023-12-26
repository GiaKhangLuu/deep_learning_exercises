import numpy as np
import math
import matplotlib.pyplot as plt
import cv2

"""
    Ex 01: Calculating sigmoid (math.exp)
"""
def basic_sigmoid(x):
    s = 1 / (1 + math.exp(-x))
    return s

"""
    Ex 02: Calculating sigmoid (numpy)
"""
def sigmoid_naive(x):
    """
    This is naive version of calculating sigmoid on an array => We have to
    loop through the array and calculating on each input 
    """
    vector_output = []
    for i in x:
        vector_output.append(basic_sigmoid(i))
    return vector_output

def sigmoid(x):
    """
    Using numpy, it will be cleaner and faster 
    """
    vector_input = np.array(x)
    vector_output = 1 / (1 + np.exp(- vector_input))
    return vector_output

"""
    Ex 03: Calculating deravative of sigmoid
"""
def sigmoid_derivative(x):
    x = np.array(x)
    s = sigmoid(x)
    ds = s * (1 - s)
    return ds

"""
    Ex 04: Image to Vector
    Method 01: flatten()
    Method 02: ravel()
    Method 03: reshape()
"""
def image2vector(image):
    """
    Assuming that image is an numpy array
    """
    # Flatten()
    v = image.flatten()
    # Ravel()
    v = image.ravel()
    # Reshape()
    v = image.reshape(-1)
    return v

"""
    Ex 06: softmax
"""
def softmax(x):
    # Converting to numpy array
    x = np.array(x)

    # Step 1: Calculating exponential func (e^x)
    x_exp = np.exp(x)

    # Step 2: Calculating sum of all elements in each row
    x_sum = np.sum(x_exp, axis=1, keepdims=True)

    # Step 3: Calculating softmax by dividing x_exp by x_sum
    s = x_exp / x_sum
    return s

def softmax_stable(x):
    """
    Softmax stable helps to solve the overflow problem when any value is huge.
    Minus each value by the maximum of each row.  
    """
    x = np.array(x)
    max_values = np.max(x, axis=1, keepdims=True)
    x_exp = np.exp(x - max_values)
    x_sum = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / x_sum


if __name__ == '__main__':
    # Run ex 01:
    #print(basic_sigmoid(1))

    # Run ex 02:
    #arr = [1, 2, 3]
    #print(sigmoid_naive(arr))

    # Run ex 03:
    #arr = [1, 2, 3]
    #print(sigmoid_derivative(arr))

    # Run ex 04: 
    #image = cv2.cvtColor(cv2.imread('./dump.jpeg'), cv2.COLOR_BGR2RGB)
    #image = np.random.rand(3, 3, 1)
    #image = cv2.resize(image, (5, 5))
    #v = image2vector(image)

    ## Plot using matplotlib
    ## Create a figure and axis
    #fig, ax = plt.subplots(ncols=2, nrows=1)

    ## Display the image data as a heatmap
    #ax[0].imshow(image, cmap='gray')
    #ax[1].imshow(np.expand_dims(v, 1), cmap='gray')

    ## Saving the plot
    #plt.savefig('./plot_ex04.png')

    ## Show the plot
    #plt.show()

    # Run ex 06:
    x = [[1, 2, 3, 0, 0], [4, 5, 6, 7, 8]]
    print(softmax(x))
    print(softmax_stable(x))