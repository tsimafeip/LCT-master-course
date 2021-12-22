import numpy as np

class NeuralNetworkModel:
    """
    A two-layer fully-connected neural network. The model has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices. The network uses a ReLU nonlinearity after the first fully
    connected layer.
    In other words, the network has the following architecture:
    input - fully connected layer - ReLU - fully connected layer - softmax
    The outputs of the second fully-connected layer are the scores for each class.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:
        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)
        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        # DO NOT CHANGE ANYTHING IN HERE

        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.
        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C. This parameter is optional; if it
          is not passed then we only return scores, and if it is passed then we
          instead return the loss and gradients.
        - reg: Regularization strength (lambda).
        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].
        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        # Compute the forward pass
        scores = 0.
        """
        Q2.2
        Perform the forward pass, computing the class probabilities for the 
        input. Store the result in the scores variable, which should be an array  
        of shape (N, C).                                                        
        """
        # TODO: START OF YOUR CODE BELOW Eq 3-7

        # N == number of samples
        # D == number of features or dimensions per sample
        
  
        # Eq 3:
        a_1 = X

        # Eq 4:
        z2 = np.add(np.matmul(a_1, W1), b1)

        # Eq 5:

        a_2 = np.maximum(0, z2)

        # Eq 6:
        z3 = np.add(np.matmul(a_2, W2), b2)

        # Eq 7:
        epsilon = 1e-9
        z3 = np.clip(z3, epsilon, 1. - epsilon)
        e_x = np.exp(z3)
        softmax = (e_x / e_x.sum(axis=1)[:,None])
        a_3 = softmax if isinstance(softmax, np.ndarray) else softmax.numpy()

        """
        DO NOT TOUCH THE CODE BELOW
        """
        try:
            assert np.all(np.isclose(np.sum(a_3, axis=1), 1.0))  # check that scores for each sample add up to 1
        except AssertionError:
            print(f'scores after softmax: \n{a_3}')
            print(f'sum of scores for all class: {np.sum(a_3, axis=1)}')

        scores = a_3

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        loss = 0.

        """
        Q2.2 Continued
        TODO: Finish the forward pass, and compute the loss. This should include
        both the data loss and L2 regularization for W1 and W2. Store the result 
        in the variable loss, which should be a scalar. Use the Softmax classifier loss.                                                          
        """
        # TODO: START OF YOUR CODE BELOW Eq 11-13
        # Implement the loss for softmax output layer

        # Eq 11
        log_probability = -np.log(a_3[np.arange(N), y])

        # Eq 12
        cross_entropy = np.sum(log_probability) / N
        

        # Eq 13
        regularization = reg*(np.sum(np.square(W1)) + np.sum(np.square(W2)))

        loss = cross_entropy + regularization

        """
        DO NOT TOUCH THE CODE BELOW
        """
        # Backward pass: compute gradients
        grads = {}
        """
        Q3.2: Compute the backward pass, computing the derivatives of the weights and biases. 
        Store the results in the grads dictionary (defined above). 
        
        For example, grads['W1'] should store the gradient on W1, and be a matrix of same size.
        """
        # TODO: START OF YOUR CODE:Backpropagation Eq 16, 18-23

        # Eq 16: gradient wrt to W2 = dj_dz3 * dz_dw
        # dj_dz3 = (1/N) * (a_3 - delta : (N,C)) shape: (N,C)
        # delta is Eq 17: only subtract 1 from where the index of a_3 corresponding to true label
        num_samples = y.shape[0]
        a_3[range(num_samples), y] -= 1.0
        dj_dz3 = a_3 / num_samples  # shape: (N,C)

        # Eq 18, 19
        dz_dw = a_2  # shape: (N,H)
        # np.dot(dj_dz3.T, dz_dw).T == np.dot(dz_dw.T, dj_dz3) (H,N) x (N,C) --> (H,C)
        dj_dw2 = np.dot(dz_dw.T, dj_dz3)  # shape: (H,C)

        # Eq 20
        # gradient wrt W2 = dj_dw2 + 2 * reg * W2
        grads['W2'] = dj_dw2 + 2 * reg * W2  # shape: (H, C)

        # Eq 21: gradients wrt to b2
        # dj_db2 = dj_dz3 * dz3_db2
        # dz3_db2 = 1
        # So, dj_db2 = dj_dz3 + ddb2(regularization term)
        # dj/db2(reg * sqr(L2-norm W1) + sqr(L2-norm W2)) = 0
        dj_db2 = dj_dz3.sum(axis=0)  # shape: (N, C) -- > (C,) same as original b2 shape
        grads['b2'] = dj_db2

        # Eq 22: gradients wrt to W1 = dj_dz3 * dz3_da2 * da2_dz2 * dz2_dW1 + 2 * reg * W1
        # dj_dz3 = (1/N) * (a_3 - delta : (N,C)) shape: (N,C)
        # dz3_da2 = W2
        # da2_dz2 = derivative_relu(z2)
        # dz2_dw1 = a_1
        def derivative_relu(u):
            return np.where(u < 0, 0.0, 1.0)

        dj_da2 = np.dot(dj_dz3, W2.T)  # shape: dj_dz3 shape: (N, C) x W2.T shape: (C,H) --> (N,H)
        dj_dz2 = dj_da2 * derivative_relu(z2)  # shape: (N,H)
        dz2_dw1 = a_1  # shape: (N, D)
        dj_dw1 = np.dot(dz2_dw1.T, dj_dz2)  # shape: dz2_dw1.T shape: (D,N) x dj_dz2 shape (N,H) --> (D, H)
        grads['W1'] = dj_dw1 + 2 * reg * W1  # shape: (D,H) + (D,H)

        # Eq 23: gradient wrt to b1
        # dj_db1 = dj_dz2 * dz2_db1
        # dz2_db1 = 1
        # So, dj_db1 = dj_dz2 + ddb1(regulrization term)
        # dj/db1(reg * sqr(L2-norm W1) + sqr(L2-norm W2)) = 0
        dj_db1 = dj_dz2.sum(axis=0)  # shape: (N,H) --> (H,) same as original b1 shape
        grads['b1'] = dj_db1

        """END OF YOUR CODE: DO NOT CHANGE THE RETURN*****"""

        return loss, grads