import numpy as np
import tensorflow as tf

OBS_DIM = 64 # Must be a multiple of 2^4 with the current model
OBS_DEPTH = 3
OBS_STACK_SIZE = 3

### Logging ###

def init_logger(lp):
    global log_path
    log_path = lp

    f = open(log_path, 'w+')
    f.close()

def log(string):
    print(string)

    with open(log_path, 'a') as f:
        f.write(string + '\n')

def gaussian_likelihood(x, mu, std):
    """Calculates the log probability of a gaussian given some input.
    Args:
        x: 2D tensor for observations drawn from the gaussian distributions.
        mu: 1D tensor for mean of the gaussian distributions.
        std: Tensor scalar for standard deviation of the gaussian distributions.
    Returns:
        1D tensor with a length equal to the number of rows in x.
        Gives the gaussian likelihood for each x for the respective gaussian distribution.
    Examples:
        >>> print([round(x) for x in discount_rewards([1, 2, 3], gamma=0.99)])
            [5.92, 4.97, 3.0]
    """
    pre_sum = -(0.5*tf.log(2.*np.pi)) - (0.5*tf.log(std)) - (tf.square(x - mu))/(2.*std+1e-8)
    
    return tf.reduce_sum(pre_sum, axis=1)