## makes the dataset for all 3x3 conway game of life kernels

import jax.numpy as jnp
from jax import grad, jit, vmap, random
    
def conway_kernel(inp):
    def c_and(a, b): return a * b
    def c_or(a, b): return a + b - a * b
    def c_eq(a, b): return jnp.maximum(0.0, 1.0 - jnp.abs(a - b)) # _/\_
    alive = inp[4]
    inp = inp.at[4].set(0)
    neighbors = jnp.sum(inp)
    return c_or(c_eq(3, neighbors), c_and(alive, c_eq(2, neighbors)))

conway_kernel_batch = lambda x: jnp.expand_dims(vmap(conway_kernel)(x), axis=-1)

def conway_draw(inp):
    out = conway_kernel(inp)
    inp = inp.reshape((3, 3))
    for row in inp:
        for x in row:
            print("x" if x > 0.5 else "-", end="")
        print(" X" if out > 0.5 else " _")

def conway_sample(key):
    return jnp.round(random.uniform(key, (9,)))

def conway_sample_batch(key, size):
    keys = random.split(key, size)
    return vmap(conway_sample)(keys)

def conway_sample_all():
    return jnp.array([[float(b) for b in bin(i)[2:].zfill(9)] for i in range(512)])

def get_conway():
    x = conway_sample_all()
    y = conway_kernel_batch(x)
    return x, y

if __name__ == '__main__':
    x, y = get_conway()

    # print(type(x))
    # print(type(y))
    print(f'x.shape: {x.shape}')
    print(f'y.shape: {y.shape}')