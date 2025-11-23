import os
import pickle

import numpy as np
import jax
import jax.numpy as jnp
from sklearn.model_selection import train_test_split

from models.jax_router import (
    TrainConfig,
    create_train_state,
    train_step,
    predict_probs,
)
from config import TRAIN_EMB_PATH, TRAIN_LABELS_PATH, CATEGORIES


# Batch DataLoader

def batch_iter(X, y, batch_size):
    n = X.shape[0]
    indices = np.arange(n)
    np.random.shuffle(indices)

    for i in range(0, n, batch_size):
        batch_idx = indices[i : i + batch_size]
        yield X[batch_idx], y[batch_idx]



# Training Loop

def train():
    print("Loading embeddings and labels")
    X = np.load(TRAIN_EMB_PATH)
    y = np.load(TRAIN_LABELS_PATH)

    print("Train-test split...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    config = TrainConfig(num_classes=len(CATEGORIES))

    print("Initializing JAX router model")
    rng = jax.random.PRNGKey(0)
    state = create_train_state(rng, input_dim=X.shape[1], config=config)

    print("Starting training...")
    for epoch in range(1, config.num_epochs + 1):
        losses = []

        for batch_x, batch_y in batch_iter(X_train, y_train, config.batch_size):
            batch_x = jnp.array(batch_x)
            batch_y = jnp.array(batch_y)

            state, loss = train_step(state, batch_x, batch_y)
            losses.append(float(loss))

        # Validation (NO JIT here, safe apply_fn usage)
        val_probs = predict_probs(
            state.params,
            state.apply_fn,
            jnp.array(X_val),
        )
        val_preds = np.argmax(np.array(val_probs), axis=-1)
        val_acc = (val_preds == y_val).mean()

        print(f"Epoch {epoch}/{config.num_epochs} | loss={np.mean(losses):.4f} | val_acc={val_acc:.4f}")


    # Save Model Parameters
   
    save_dir = "models/saved_params"
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, "router.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(
            {
                "params": state.params,
                "input_dim": X.shape[1],
                "config": config,
                "categories": CATEGORIES,
            },
            f,
        )

    print("\nRouter model saved successfully!")
    print("Saved at:", save_path)



# Main

if __name__ == "__main__":
    train()
