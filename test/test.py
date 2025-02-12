###shows the difference in loss between using jnp.where() and boolean indexing

import jax
import jax.numpy as jnp

def binary_cross_entropy(prediction, target):
    return -(target * jnp.log(prediction) + (1 - target) * jnp.log(1 - prediction))


preds = jax.random.uniform(jax.random.PRNGKey(0), (100, )).reshape((-1,1)) #predictions
LABELS = jax.random.uniform(jax.random.PRNGKey(0), (100, )).reshape((-1,1)) #the annotated labels y

pair_lab = LABELS - LABELS.T
pair_pred = preds - preds.T

ptk = jnp.abs(pair_lab) > 0.25

pair_labw = jnp.where(ptk, pair_lab, -jnp.nan)
pair_labm = pair_lab[ptk]

pair_labw = jnp.where(pair_labw > 0, True, False)
pair_labm = jnp.where(pair_labm > 0, True, False)

pair_predw = jnp.where(ptk, pair_pred, -jnp.nan)
pair_predm = pair_pred[ptk]

pair_predw = jax.nn.sigmoid(pair_predw)
pair_predm = jax.nn.sigmoid(pair_predm)

lossw = binary_cross_entropy(pair_predw, pair_labw)
lossm = binary_cross_entropy(pair_predm, pair_labm)

print("loss using jnp.where()", jnp.mean(lossw, where=~jnp.isnan(lossw)))
print("loss using boolean indexing", jnp.mean(lossm, where=~jnp.isnan(lossm)))