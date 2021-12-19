
import numpy as np

#######################################################
# DO NOT MODIFY ANY CODE OTHER THAN THOSE TODO BLOCKS #
#######################################################

def binary_train(X, y, loss="perceptron", w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: binary training labels, a N dimensional numpy array where 
    N is the number of training points, indicating the labels of 
    training data (either 0 or 1)
    - loss: loss type, either perceptron or logistic
	- w0: initial weight vector (a numpy array)
	- b0: initial bias term (a scalar)
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: D-dimensional vector, a numpy array which is the final trained weight vector
    - b: scalar, the final trained bias term

    Find the optimal parameters w and b for inputs X and y.
    Use the *average* of the gradients for all training examples
    multiplied by the step_size to update parameters.	
    """
    N, D = X.shape
    assert len(np.unique(y)) == 2


    w = np.zeros(D)
    if w0 is not None:
        w = w0

    b = 0
    if b0 is not None:
        b = b0

    X_dash = np.insert(X, 0, 1, axis=1)
    w_dash = np.insert(w, 0, b, axis=0)
    y = np.where( y= =0, -1, 1)

    if loss == "perceptron":
        ################################################
        # TODO 1 : perform "max_iterations" steps of   #
        # gradient descent with step size "step_size"  #
        # to minimize perceptron loss (use -1 as the   #
        # derivative of the perceptron loss at 0)      #
        ################################################
        for t in range(max_iteration s +1):
            # print("train:Perceptron")
            y_t = binary_predict(X_dash, w_dash, 10101010)  # prediction
            y_t = np.where(y_ t= =0, -1, 1) # set to -1 where 0
            # print(y_t)
            y_wt_x = y * y_t # actualY * predictedY
            y_wt_x = np.where(y_wt_x <= 0, 1, 0)
            # print(y_wt_x)
            y_ind = y_wt_x * y  # indicator
            y_ind_x = np.dot(y_ind, X_dash)
            # print(y_ind_x)
            e = step_siz e /N
            # print("here2")
            sg = e * y_ind_x
            w_dash = np.add(w_dash, sg)  # 3final update
            # print(w_dash)




    elif loss == "logistic":
        ################################################
        # TODO 2 : perform "max_iterations" steps of   #
        # gradient descent with step size "step_size"  #
        # to minimize logistic loss                    # 
        ################################################
        for i in range(max_iteration s +1):
            # print("train:Log")
            y_wt_x = y * np.dot(X_dash, w_dash)
            # print(y_wt_x)
            sig = sigmoid(-y_wt_x)
            sig = sig * y
            sig = np.dot(sig, X_dash)
            e = step_siz e /N
            w_dash = w_dash + e* sig
            # print(w_dash)



    else:
        raise "Undefined loss function."

    b = w_dash[0]
    w = np.delete(w_dash, 0)
    assert w.shape == (D,)
    return w, b


def sigmoid(z):
    """
    Inputs:
    - z: a numpy array or a float number

    Returns:
    - value: a numpy array or a float number after applying the sigmoid function 1/(1+exp(-z)).
    """

    ############################################
    # TODO 3 : fill in the sigmoid function    #
    ############################################
    # print(1 / (1+np.exp(-z)))
    return (1 / (1 + np.exp(-z)))


def binary_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of your learned model
    - b: scalar, which is the bias of your model

    Returns:
    - preds: N-dimensional vector of binary predictions (either 0 or 1)
    """
    N, D = X.shape

    #############################################################
    # TODO 4 : predict DETERMINISTICALLY (i.e. do not randomize)#
    #############################################################
    if b != 10101010:
        # print("bin_pred")
        X = np.insert(X, 0, 1, axis=1)
        w = np.insert(w, 0, b, axis=0)

    y_t = np.dot(X, w)
    preds = np.sign(y_t)
    # print(preds)
    preds = np.where(preds == -1, 0, 1)
    # print(preds)
    assert preds.shape == (N,)
    return preds


def multiclass_train(X, y, C,
                     w0=None,
                     b0=None,
                     gd_type="sgd",
                     step_size=0.5,
                     max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of 
    training data (0, 1, ..., C-1)
    - C: number of classes in the data
    - gd_type: gradient descent type, either GD or SGD
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform (stochastic) gradient descent

    Returns:
    - w: C-by-D weight matrix, where C is the number of classes and D 
    is the dimensionality of features.
    - b: a bias vector of length C, where C is the number of classes

    Implement multinomial logistic regression for multiclass 
    classification. Again for GD use the *average* of the gradients for all training 
    examples multiplied by the step_size to update parameters.

    You may find it useful to use a special (one-hot) representation of the labels, 
    where each label y_i is represented as a row of zeros with a single 1 in
    the column that corresponds to the class y_i. Also recall the tip on the 
    implementation of the softmax function to avoid numerical issues.
    """

    N, D = X.shape

    w = np.zeros((C, D))
    if w0 is not None:
        w = w0

    b = np.zeros(C)
    if b0 is not None:
        b = b0

    np.random.seed(42)  # DO NOT CHANGE THE RANDOM SEED IN YOUR FINAL SUBMISSION
    X_dash = np.append(X, np.ones((N, 1)), axis=1)
    w_dash = np.append(w, np.array([b]).T, axis=1)
    if gd_type == "sgd":

        for it in range(max_iterations):
            n = np.random.choice(N)
            ####################################################
            # TODO 5 : perform "max_iterations" steps of       #
            # stochastic gradient descent with step size       #
            # "step_size" to minimize logistic loss. We already#
            # pick the index of the random sample for you (n)  #
            ####################################################
            # print("train:SGD")
            w_t = np.matmul(w_dash, X_dash[n].T)
            num = np.subtract(w_t, np.amax(w_t, axis=0))
            # print(num)
            num = [np.exp(i) for i in num]
            d = np.sum(num);
            probability = [i / d for i in num]
            probability[y[n]] -= 1
            sgd = np.matmul(np.array([probability]).T, np.array([X_dash[n]]))
            w_dash = w_dash - step_size * sgd
            # print(w_dash)



    elif gd_type == "gd":
        ####################################################
        # TODO 6 : perform "max_iterations" steps of       #
        # gradient descent with step size "step_size"      #
        # to minimize logistic loss.                       #
        ####################################################
        y_id = np.eye(C)[y]  # converting to required dimensions
        for i in range(max_iterations):
            # print("train:SGD")
            w_t = X_dash.dot(w_dash.T)
            num = np.exp(w_t - np.amax(w_t))
            d = np.sum(num, axis=1)
            y_t = (num.T / d).T
            y_t = y_t - y_id
            gd = np.dot(y_t.T, X_dash)
            w_dash = w_dash - (step_size / N) * gd
            # print(w_dash)



    else:
        raise "Undefined algorithm."

    b = w_dash[:, -1]
    w = np.delete(w_dash, -1, 1)
    assert w.shape == (C, D)
    assert b.shape == (C,)

    return w, b


def multiclass_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: weights of the trained model, C-by-D 
    - b: bias terms of the trained model, length of C

    Returns:
    - preds: N dimensional vector of multiclass predictions.
    Predictions should be from {0, 1, ..., C - 1}, where
    C is the number of classes
    """
    N, D = X.shape
    #############################################################
    # TODO 7 : predict DETERMINISTICALLY (i.e. do not randomize)#
    #############################################################
    # print("pred:MC")
    w_dash = np.insert(w, 0, b, axis=1)
    X_dash = np.insert(X, 0, 1, axis=1)
    w_t_x = np.dot(X_dash, w_dash.T)
    preds = np.argmax(w_t_x, axis=1)

    assert preds.shape == (N,)
    return preds




