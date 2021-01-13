import jax
import jax.numpy as jnp
from flax import optim
from flax import linen as nn
from flax import jax_utils
import numpy as np
from flax.training import common_utils

from util import load_train_val_test, batchify, batchify_test, DataLoader
from layers import Transformer


def create_learning_rate_scheduler(base_learning_rate=0.5, warmup_steps=8000):
    """Define our learning rate schedule."""
    def step_fn(step):
        return jnp.asarray(
            base_learning_rate * 
            jnp.minimum(1.0, step / warmup_steps) /
            jnp.sqrt(jnp.maximum(step, warmup_steps)), dtype=jnp.float32)
    return step_fn


def compute_weighted_cross_entropy(logits,
                                   targets,
                                   weights=None,
                                   label_smoothing=0.0):
    """Compute weighted cross entropy and entropy for log probs and targets.
    Args:
    logits: [batch, length, num_classes] float array.
    targets: categorical targets [batch, length] int array.
    weights: None or array of shape [batch, length].
    label_smoothing: label smoothing constant, used to determine the on and off
     values.
    Returns:
    Tuple of scalar loss and batch normalizing factor.
    """
    if logits.ndim != targets.ndim + 1:
        raise ValueError(f"Incorrect shapes. Got shape {logits.shape} logits and {targets.shape} targets")
    vocab_size = logits.shape[-1]
    confidence = 1.0 - label_smoothing
    low_confidence = (1.0 - confidence) / (vocab_size - 1)
    normalizing_constant = -(
      confidence * jnp.log(confidence) +
      (vocab_size - 1) * low_confidence * jnp.log(low_confidence + 1e-20))
    soft_targets = common_utils.onehot(
      targets, vocab_size, on_value=confidence, off_value=low_confidence)

    loss = -jnp.sum(soft_targets * nn.log_softmax(logits), axis=-1)
    loss = loss - normalizing_constant

    normalizing_factor = np.prod(targets.shape)
    if weights is not None:
        loss = loss * weights
        normalizing_factor = weights.sum()

    return loss.sum(), normalizing_factor


def compute_weighted_accuracy(logits, targets, weights=None):
    """Compute weighted accuracy for log probs and targets.

    Args:
    logits: [batch, length, num_classes] float array.
    targets: categorical targets [batch, length] int array.
    weights: None or array of shape [batch x length]

    Returns:
    Tuple of scalar accuracy and batch normalizing factor.
    """
    if logits.ndim != targets.ndim + 1:
        raise ValueError(f"Incorrect shapes. Got shape {logits.shape} logits and {targets.shape} targets")
    loss = jnp.equal(jnp.argmax(logits, axis=-1), targets)
    normalizing_factor = np.prod(logits.shape[:-1])
    if weights is not None:
        loss = loss * weights
        normalizing_factor = weights.sum()

    return loss.sum() / normalizing_factor


# +
seed = 42

rng = jax.random.PRNGKey(seed)
rng, init_rng = jax.random.split(rng)
_, dropout_rng = jax.random.split(rng)

# +
embed_size = 50
num_heads = 1
N = 2
# max_seq_len = 20

input_shape = (batch_size, max_seq_len)
model = Transformer(num_tokens=num_items+1, 
                          embed_size=embed_size, 
                          max_seq_len=max_seq_len, 
                          N=N, 
                          num_heads=num_heads, 
                          causal=True, attention_dropout=0.2, ff_dropout=0.2)
# -

init_vars = jax.jit(model.init)({"params": init_rng, "dropout": dropout_rng}, x=jnp.ones(input_shape, jnp.int32))

# +
base_lr = 2.0

learning_rate_fn = create_learning_rate_scheduler(base_lr, warmup_steps=4000)

# apply an optimizer to this tree
optimizer_def = optim.Adam(
  base_lr,
  beta1=0.9,
  beta2=0.98,
  eps=1e-9)

optimizer = optimizer_def.create(init_vars["params"])
# -

train_ds, valid_ds, test_ds, num_users, num_items = load_train_val_test("Video")
train_dl = DataLoader(train_ds, batch_size, max_seq_len, seed = 42)

# +
start_step = 0
num_train_steps = 50000      # Max number of training steps.
train_loss = 0. ; train_acc = 0.
log_every = 200
do_eval = 1000
train_it = iter(train_dl)
start = time.time()

for step in range(start_step, num_train_steps):
    try:
        seq, pos = next(train_it)
    except StopIteration:
        train_it = iter(train_dl)
        seq, pos = next(train_it)
    inputs = jnp.array(seq, dtype=jnp.int32)
    targets = jnp.array(pos, dtype=jnp.int32)

    # Core training step.
    # batch = common_utils.shard(jax.tree_map(lambda x: x._numpy(), batch))
    # optimizer, metrics = train_step(model, optimizer, inputs, targets, dropout_rng=dropout_rng)
    
    dropout_rng = jax.random.fold_in(dropout_rng, optimizer.state.step)
    weights = targets > 0 # ignore padding

    def loss_fn(params):
        """loss function used for training."""
        logits = model.apply({"params": params}, inputs, rngs={"dropout": dropout_rng})
        loss, weight_sum = compute_weighted_cross_entropy(logits, targets, weights, label_smoothing=0.)
        mean_loss = loss / weight_sum
        return mean_loss, logits

    lr = learning_rate_fn(optimizer.state.step)
    # has_aux: Indicates whether ``fun`` returns a pair where the
    # first element is considered the output of the mathematical function to be
    # differentiated and the second element is auxiliary data. Default False.
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grad = grad_fn(optimizer.target) # this plays the role of params
    
    # grad = jax.lax.pmean(grad, "batch")
    optimizer = optimizer.apply_gradient(grad, learning_rate=lr)

    # compute acc
    acc = compute_weighted_accuracy(logits, targets, weights)
    
    # train_loss += metrics["loss"]
    # train_acc += metrics["acc"]
    train_loss += loss
    train_acc += acc
    
    if step % log_every == 0 and step > 0:
        hit10 = compute_hitk(logits, targets, topk = 10)
        elapsed = time.time() - start
        
        print(f"[{step}][LR:{lr:.4f}] Loss: {train_loss / log_every:.4f} - Acc: {train_acc / log_every:.4f} - Hit10: {hit10:.4f}")
        print(f"Elapsed: {elapsed:.2f}")
        train_loss = 0. ; train_acc = 0. ; start = time.time()

    if step % do_eval == 0 and step > 0:
        # Eval
        valid_dl = batchify_test(valid_ds, batch_size=batch_size, max_seq_len=max_seq_len)
        valid_acc = []
        valid_hitk = []
        for seq, pos in valid_dl:
            inputs = jnp.array(seq, dtype=jnp.int32)
            targets = jnp.array(pos, dtype=jnp.int32)
            logits = model.apply({"params": optimizer.target}, inputs, deterministic=True)
            inputs_len = (inputs > 0).sum(-1)
            logits_last = logits[jnp.arange(logits.shape[0]), inputs_len - 1, :] # (b, v)
            logits_topk = logits_last.argsort(-1)[:, -10:] # (b, 10)
            y_pred = logits_last.argmax(-1)
            valid_hitk += [label in topk for label, topk in zip(targets, logits_topk)]
            valid_acc.extend(y_pred == targets)
        print(f"[Val] Hit10: {jnp.array(valid_hitk).mean():.4f} - Acc: {jnp.array(valid_acc).mean():.4f}")
