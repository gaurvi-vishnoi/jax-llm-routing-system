from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax


# -----------------------------------------------------------
# Training configuration
# -----------------------------------------------------------
@dataclass
class TrainConfig:
    learning_rate: float = 1e-3
    hidden_dim: int = 128
    num_classes: int = 5
    batch_size: int = 64
    num_epochs: int = 8
    weight_decay: float = 1e-4


# -----------------------------------------------------------
# Router model (rename Router → RouterModel)
# -----------------------------------------------------------
class RouterModel(nn.Module):
    hidden_dim: int
    num_classes: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.num_classes)(x)
        return x


# -----------------------------------------------------------
# Create Train State
# -----------------------------------------------------------
def create_train_state(rng, input_dim: int, config: TrainConfig) -> train_state.TrainState:
    model = RouterModel(hidden_dim=config.hidden_dim, num_classes=config.num_classes)

    params = model.init(rng, jnp.ones((1, input_dim)))["params"]

    tx = optax.adamw(
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )


# -----------------------------------------------------------
# Loss function
# -----------------------------------------------------------
def cross_entropy_loss(logits, labels):
    one_hot = jax.nn.one_hot(labels, logits.shape[-1])
    return optax.softmax_cross_entropy(logits=logits, labels=one_hot).mean()


# -----------------------------------------------------------
# Train step (JIT compiled)
# -----------------------------------------------------------
@jax.jit
def train_step(state: train_state.TrainState, batch_x, batch_y):
    def loss_fn(params):
        logits = state.apply_fn({"params": params}, batch_x)
        loss = cross_entropy_loss(logits, batch_y)
        return loss, logits

    (loss, _), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


# -----------------------------------------------------------
# Prediction helpers
# -----------------------------------------------------------
def predict_logits(params, apply_fn, x):
    return apply_fn({"params": params}, x)


def predict_probs(params, apply_fn, x):
    logits = predict_logits(params, apply_fn, x)
    return jax.nn.softmax(logits, axis=-1)
