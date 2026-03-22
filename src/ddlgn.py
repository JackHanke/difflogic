## deep differentiable logic gate network implementation
import itertools

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, random


GATES = 16

def gate_all(a, b):
    return jnp.array([
        # jnp.maximum(0., a)
        jnp.zeros_like(a),
        a * b,
        a - a*b,
        a,
        b - a*b,
        b,
        a + b - 2.0*a*b,
        a + b - a*b,
        1.0 - (a + b - a*b),
        1.0 - (a + b - 2.0*a*b),
        1.0 - b,
       	1.0 - b + a*b,
        1.0 - a,
        1.0 - a + a*b,
        1.0 - a*b,
        jnp.ones_like(a),
    ])

# gate_all(left, right) and w have shape (16, n)
# where n is the dimension of left/right
# This is a batched dot product along the second axis (axis 1)
def gate(left, right, w, hard):
    w_gate = \
        jnp.exp(w) / jnp.sum(jnp.exp(w), axis=0, keepdims=True) if not hard \
        else jax.nn.one_hot(jnp.argmax(w, axis=0), GATES).T
    return jnp.sum(gate_all(left, right) * w_gate, axis=0)

# def relu(left, right, w):
#     return jnp.maximum(0., left)

def gate_normalize(w):
    sum_col = jnp.sum(w, axis=0)
    return w / sum_col[None,:]

# uniform random vectors length 16 whose entries sum to 1
def rand_gate(key, n):
    # return gate_normalize(random.uniform(key, (GATES, n)))
    result = jnp.zeros((GATES, n))
    result = result.at[3, :].set(10.0)
    return result
    # return jnp.full((GATES, n), 1. / GATES)

def pairs_rand(key, m, n_left):
    keys = random.split(key, n_left)
    return jnp.stack([random.permutation(key, m)[:2] for key in keys]).T

def pairs_rand_pad(key, pairs, m, n):
    if len(pairs) < n:
        print("↳ with random padding")
        pairs_pad = pairs_rand(key, m, n - len(pairs)).T
        pairs = jnp.concatenate([pairs, pairs_pad], axis=0)
    else:
        pairs = pairs[:n]
    return pairs

def pairs_comb_pad(key, pairs, m, n):
    def canonical(t): return tuple(sorted([*t]))
    existing = set(map(canonical, pairs.tolist()))
    if len(pairs) < n:
        perms = random.permutation(key, jnp.array(list(itertools.combinations(list(range(m)), 2))))
        pairs_new = jnp.array([p for p in perms if canonical(p.tolist()) not in existing])
        pairs_rand_new = pairs_rand(key, m, max(0, n - len(pairs) - len(pairs_new))).T
        print("↳ with comb padding", len(pairs_new))
        print("↳ with rand padding", len(pairs_rand_new))
        pairs = jnp.concatenate([pairs, pairs_new, pairs_rand_new], axis=0)
    else:
        pairs = pairs[:n]
    return pairs

def wire_from_pairs(pairs, m, n):
    assert n, len(pairs)
    left = jax.nn.one_hot(pairs[0, :], num_classes=m)
    right = jax.nn.one_hot(pairs[1, :], num_classes=m)
    return left, right

def wire_rand(key, m, n):
    pairs = pairs_rand(key, m, n)
    return wire_from_pairs(pairs, m, n)

def wire_rand_unique(key, m, n):
    print("unique wire", m, "→", n)
    evens = zip(range(0, m)[0::2], range(0, m)[1::2])
    odds = zip(range(0, m)[1::2], list(range(0, m)[2::2]) + [0])
    pairs = jnp.array([*evens, *odds])
    key_rand, key_perm = random.split(key)
    pairs = pairs_comb_pad(key_rand, pairs, m, n)
    pairs = random.permutation(key_perm, pairs)
    return wire_from_pairs(pairs.T, m, n)

# m input, n output
def wire_rand_comb(key, m, n):
    print("comb wire", m, "→", n)
    key, key_sub = random.split(key)
    pairs = random.permutation(key_sub, jnp.array(list(itertools.combinations(list(range(m)), 2))))
    pairs = pairs_rand_pad(key, pairs, m, n)
    return wire_from_pairs(pairs.T, m, n)

def wire_tree(m, n):
    print("tree wire", m, "→", n)
    assert m, 2*n
    pairs = jnp.arange(m).reshape((2, n))
    return wire_from_pairs(pairs, m, n)

def rand_layer(key, m, n):
    left_key, right_key, gate_key = random.split(key, 3)
    if n > m:
        left, right = wire_rand_comb(left_key, m, n)
    else:
        left, right = wire_rand_unique(left_key, m, n)
    param = rand_gate(gate_key, n)
    wires = (left, right)
    return param, wires

def rand_network(key, sizes):
    keys = random.split(key, len(sizes))
    dims = zip(keys, sizes[:-1], sizes[1:])
    return list(zip(*[rand_layer(*dim) for dim in dims]))

def predict(params, wires, inp, hard):
    active = inp
    for param, (left, right) in zip(params, wires):
        outs_l = jnp.dot(left, active)
        outs_r = jnp.dot(right, active)
        active = gate(outs_l, outs_r, param, hard)
    return active

# TODO Aggregation of Output Neurons

# TODO what is this
predict_batch = vmap(predict, in_axes=(None, None, 0, None))
