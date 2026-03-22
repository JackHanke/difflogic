## train ddlgn
import time
import optax
import functools

import jax.numpy as jnp
from jax import grad, jit, vmap, random

from src.ddlgn import *
from src.extract import *
from datasets.conway import get_conway
from datasets.ttt import get_ttt

# l2 loss
def loss(params, wires, inp, out, hard):
    preds = predict_batch(params, wires, inp, hard)
    return jnp.mean(jnp.square(preds - out))

# TODO CSE

@functools.partial(jit, static_argnums=(4,))
def update_adamw(params, wires, x, y, opt, opt_state):
    # I think params has to be the first argument?
    grads = grad(loss)(params, wires, x, y, False)
    grads, opt_state = opt.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, grads)
    return new_params, opt_state


def train_adamw(
        x,
        y,
        key, 
        params, 
        wires, 
        epochs: int = 3001, 
        batch_size: int = 512
    ):
    '''  '''
    
    opt = optax.chain(
        optax.clip(100.0),
        optax.adamw(learning_rate=0.05, b1=0.9, b2=0.99, weight_decay=1e-2)
    )
    opt_state = opt.init(params)
    
    keys = random.split(key, epochs)
    for (i, key_epoch) in enumerate(keys):
        key_train, key_accuracy = random.split(key_epoch)

        time_start = time.time()
        params, opt_state = update_adamw(params, wires, x, y, opt, opt_state)
        time_epoch = time.time() - time_start

        print(f"Epoch ({i+1}/{epochs}) in {time_epoch:.3g} s/epoch", end="   \r")
        if i % 1000 == 0: debug_loss(key_accuracy, params, wires, x, y)
        # if i % 10000 == 0: debug_params(params)
    return params

def debug_loss(key, params, wires, x, y):
    print()

    # TODO figure out why he gets the dataset again
    # x_test, y_test = get_conway()
    x_test, y_test = get_ttt()

    train_loss = loss(params, wires, x, y, False)
    test_loss = loss(params, wires, x_test, y_test, False)
    test_loss_hard = loss(params, wires, x_test, y_test, True)

    preds = predict_batch(params, wires, x_test, False)
    preds_hard = predict_batch(params, wires, x_test, True)
    print("[", *[f"{x:.3g}" for x in preds[0:5].flatten().tolist()], "]", preds_hard[0:5].flatten(), y_test[0:5].flatten())
    print(f"train_loss: {train_loss:.3g}", end="; ")
    print(f"test_loss: {test_loss:.3g}", end="; ")
    print(f"test_loss_hard: {test_loss_hard:.3g}")

def debug_params(params):
    for i, param in enumerate(params):
        print("LAYER", i, param.shape)
        for gate in param.T.tolist():
            for logic in gate:
                if logic > 1: print("█ ", end="")
                elif logic < 0: print("▁ ", end="")
                else: print("▄ ", end="")
            print()

if __name__ == "__main__":

    SEED = 379009
    key = random.PRNGKey(SEED)
    param_key, train_key = random.split(key)

    # init dataset
    # x, y = get_conway()
    x, y = get_ttt()

    # init network
    layer_sizes = [9, *([128] * 17), 64, 32, 16, 8, 4, 2, 1]
    params, wires = rand_network(param_key, layer_sizes)

    # train model
    params_trained = train_adamw(
        x,
        y,
        key=train_key, 
        params=params, 
        wires=wires, 
        epochs=3001, 
        batch_size=512
    )

    # compile model
    ext_compile_to_c(params_trained, wires)

    # for instr in ext_logic(params, wires):
    #     print(ext_format(instr), end="")
