## create jax array of perfect tic tac toe gameplay dataset, encoded to binary
import numpy as np
import jax.numpy as jnp
    
def get_ttt():
    ''' reads binary dataset from csv, returns jax arrays '''
    # path = './datasets/ttt_binary.csv'
    path = './datasets/ttt_best_bin.csv'
    data = np.loadtxt(path, delimiter=',')
    x, y = data[1:, :18], data[1:, 18:] # skip header
    # print(data)
    x, y = jnp.array(x), jnp.array(y, dtype='int32')
    y = y.flatten()
    return x, y

if __name__ == '__main__':
    x, y = get_ttt()

    print(f'x.shape, type: {x.shape} {x.dtype}')
    print(f'y.shape, type: {y.shape} {y.dtype}')