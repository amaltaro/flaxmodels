import tensorflow as tf
#import tensorflow_datasets as tfds
import jax
import jax.numpy as jnp
from jax.lib import xla_bridge
import flax
#from flax.optim import dynamic_scale as dynamic_scale_lib
import flax.linen as nn
from flax.training import train_state
from flax.training import common_utils
from flax.training import checkpoints
from flax.training import lr_schedule
import optax
import numpy as np
import dataclasses
import functools
from tqdm import tqdm
from typing import Any
import argparse
import wandb
import os
import logging
import sys
import time
import socket
sys.path.append("../..")
import flaxmodels as fm
import json
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch


def setupLogger():
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(module)s:%(message)s")
    handler.setFormatter(formatter)
    root.addHandler(handler)
    return root


def cross_entropy_loss(logits, labels):
    """
    Computes the cross entropy loss.

    Args:
        logits (tensor): Logits, shape [B, num_classes].
        labels (tensor): Labels, shape [B,].

    Returns:
        (tensor): Cross entropy loss, shape [].
    """
    return -jnp.sum(common_utils.onehot(labels, num_classes=logits.shape[1]) * logits) / labels.shape[0]


def compute_metrics(logits, labels):
    """
    Computes the cross entropy loss and accuracy.

    Args:
        logits (tensor): Logits, shape [B, num_classes].
        labels (tensor): Labels, shape [B,].

    Returns:
        (dict): Dictionary containing the cross entropy loss and accuracy.
    """
    loss = cross_entropy_loss(logits, labels)
    accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
    metrics = {'loss': loss, 'accuracy': accuracy}
    return metrics


class TrainState(train_state.TrainState):
    """
    Simple train state for the common case with a single Optax optimizer.

    Attributes:
        dynamic_scale (dynamic_scale_lib.DynamicScale): Dynamic loss scaling for mixed precision gradients.
        epoch (int): Current epoch.
    """
    #dynamic_scale: dynamic_scale_lib.DynamicScale
    epoch: int


def restore_checkpoint(state, path):
    """
    Restores checkpoint with best validation score.

    Args:
        state (train_state.TrainState): Training state.
        path (str): Path to checkpoint.

    Returns:
        (train_state.TrainState): Training state from checkpoint.
    """
    return checkpoints.restore_checkpoint(path, state)


def save_checkpoint(state, step_or_metric, path):
    """
    Saves a checkpoint from the given state.

    Args:
        state (train_state.TrainState): Training state.
        step_or_metric (int of float): Current training step or metric to identify the checkpoint.
        path (str): Path to the checkpoint directory.

    """
    if jax.process_index() == 0:
        state = jax.device_get(jax.tree_map(lambda x: x[0], state))
        checkpoints.save_checkpoint(path, state, step_or_metric, keep=3)




def train_step(state, batch, rng):
    def loss_fn(params):
        logits = state.apply_fn(params,
                                batch['image'],
                                rngs={'dropout': rng})
        loss = cross_entropy_loss(logits, batch['label'])
        return loss, logits

    #dynamic_scale = state.dynamic_scale

    #if dynamic_scale:
    #    grad_fn = dynamic_scale.value_and_grad(loss_fn, has_aux=True, axis_name='batch')
    #    dynamic_scale, is_fin, aux, grads = grad_fn(state.params)
        # dynamic loss takes care of averaging gradients across replicas
    #else:
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    aux, grads = grad_fn(state.params)
    # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
    grads = jax.lax.pmean(grads, axis_name='batch')

    logits = aux[1]
    metrics = compute_metrics(logits, batch['label'])

    new_state = state.apply_gradients(grads=grads)

   # if dynamic_scale:
    #    # if is_fin == False the gradients contain Inf/NaNs and optimizer state and
    #    # params should be restored (= skip this step).
    #    new_state = new_state.replace(opt_state=jax.tree_multimap(functools.partial(jnp.where, is_fin),
    #                                                              new_state.opt_state,
    #                                                              state.opt_state),
    #                                  params=jax.tree_multimap(functools.partial(jnp.where, is_fin),
    #                                                           new_state.params,
    #                                                           state.params))
    #    metrics['scale'] = dynamic_scale.scale

    return new_state, metrics


def eval_step(state, batch):
    logits = state.apply_fn(state.params, batch['image'], train=False, mutable=False)
    return compute_metrics(logits, batch['label'])


def train_and_evaluate(config):
    """
    Runs all the heavy processing
    """
    logger = setupLogger()
    totalStart = time.time()
    logger.info(f"Starting train_and_evaluate method at time: {totalStart}")
    num_devices = jax.device_count()
    logger.info(f"Hostname: {socket.gethostname()} using {num_devices} GPU devices")
    logger.info(f"Local devices: {jax.local_devices()}")
    logger.info(f"Environment info: {jax.print_environment_info()}")


    #--------------------------------------
    # Data
    #--------------------------------------
    logger.info(f"Configuring dataloader")

    traindir = os.path.join(config.data, 'train')
    valdir = os.path.join(config.data, 'val')


    # Define the training dataset
    # NOTE: we do not normalize images at the pre-processing step
    # because it's already embedded in the Jax model
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]))

    # Define the validation dataset
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]))

    # NOTE: DistributedSampler is only used when running distributed training,
    # hence using GPUs in multiple nodes.
    if False:  # Just so we remain consistent with the PyTorch script implementation
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
    else:
        train_sampler = None
        val_sampler = None

    gpu_batch_size = config.batch_size // num_devices
    logger.info(f"Actual GPU batch size set to: {gpu_batch_size}")
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=gpu_batch_size, shuffle=(train_sampler is None),
        num_workers=config.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=gpu_batch_size, shuffle=False,
        num_workers=config.workers, pin_memory=True, sampler=val_sampler)

    dataset_size = len(train_loader)


    #--------------------------------------
    # Seeding, Devices, and Precision
    #--------------------------------------
    rng = jax.random.PRNGKey(config.random_seed)

    if config.mixed_precision:
        dtype = jnp.float16
    else:
        dtype = jnp.float32

    #platform = jax.local_devices()[0].platform
    #if config.mixed_precision and platform == 'gpu':
    #    dynamic_scale = dynamic_scale_lib.DynamicScale()
    #else:
    #    dynamic_scale = None


    #--------------------------------------
    # Initialize Models
    #--------------------------------------
    logger.info(f"Initializing models for config.arch: {config.arch}")
    rng, init_rng, init_rng_dropout = jax.random.split(rng, num=3)

    if config.arch == 'vgg16':
        model = fm.VGG16(output='log_softmax', pretrained=None, num_classes=config.num_classes, dtype=dtype)
    elif config.arch == 'vgg19':
        model = fm.VGG19(output='log_softmax', pretrained=None, num_classes=config.num_classes, dtype=dtype)

    init_rngs = {'params': init_rng, 'dropout': init_rng_dropout}
    params = model.init(init_rngs, jnp.ones((1, config.img_size, config.img_size, config.img_channels), dtype=dtype))

    #--------------------------------------
    # Initialize Optimizer
    #--------------------------------------
    steps_per_epoch = dataset_size // config.batch_size

    logger.info(f"Initializing Optimizer with steps_per_epoch: {steps_per_epoch}")

    #learning_rate_fn = lr_schedule.create_cosine_learning_rate_schedule(config.learning_rate,
    #                                                                    steps_per_epoch,
    #                                                                    config.num_epochs - config.warmup_epochs,
    #                                                                    config.warmup_epochs)

    tx = optax.adam(learning_rate=config.learning_rate)  #tx = optax.adam(learning_rate=learning_rate_fn)

    state = TrainState.create(apply_fn=model.apply,
                              params=params,
                              tx=tx,
                              #dynamic_scale=dynamic_scale,
                              epoch=0)

    step = 0
    epoch_offset = 0
    if config.resume:
        ckpt_path = checkpoints.latest_checkpoint(config.ckpt_dir)
        state = restore_checkpoint(state, ckpt_path)
        step = jax.device_get(state.step)
        epoch_offset = jax.device_get(state.epoch)

    state = flax.jax_utils.replicate(state)

    #--------------------------------------
    # Create train and eval steps
    #--------------------------------------
    logger.info(f"Create train and eval steps")

    p_train_step = jax.pmap(functools.partial(train_step), axis_name='batch')
    p_eval_step = jax.pmap(eval_step, axis_name='batch')

    #--------------------------------------
    # Training
    #--------------------------------------

    best_val_acc = 0.0
    ### Time summary will be dumped as a json at ckpt_dir in the format of
    timeSummary = []
    i = 0
    for epoch in range(epoch_offset, config.num_epochs):
        logger.info(f"Starting training for epoch number: {epoch}")
        #thisEpoch = {"epoch_num": 0, "epoch_time": 0.0, "valid_time": 0.0, "train_time": 0.0}
        thisEpoch = {"epoch_num": epoch, "epoch_time": 0.0, "valid_time": 0.0, "train_time": 0.0,
                     'validation/accuracy': 0, 'training/accuracy': 0}
        #pbar = tqdm(total=dataset_size)

        accuracy = 0.0
        n = 0
        epochStart = time.time()

        for image, label in train_loader:
            image = image.numpy().astype(dtype)
            image = jnp.moveaxis(image,(0,2,3,1),(0,1,2,3))
            #pbar.update(num_devices * config.batch_size)
            label = label.numpy().astype(dtype)

            if image.shape[0] % num_devices != 0:
                # Batch size must be divisible by the number of devices
                continue

            # Reshape images from [num_devices * batch_size, height, width, img_channels]
            # to [num_devices, batch_size, height, width, img_channels].
            # The first dimension will be mapped across devices with jax.pmap.
            image = jnp.reshape(image, (num_devices, -1) + image.shape[1:])
            label = jnp.reshape(label, (num_devices, -1) + label.shape[1:])

            rng, _ = jax.random.split(rng)
            rngs = jax.random.split(rng, num=num_devices)
            state, metrics = p_train_step(state, {'image': image, 'label': label}, rng=rngs)
            accuracy += metrics['accuracy']
            n += 1
            step += 1

        #pbar.close()
        accuracy /= n
        # update training time and accuracy for this epoch
        thisEpoch["training/accuracy"] = jnp.mean(metrics['accuracy']).item()
        thisEpoch["train_time"] = round(time.time() - epochStart, 3)

        print(f'Epoch: {epoch}')
        print('Training accuracy:', jnp.mean(accuracy))

        #--------------------------------------
        # Validation
        #--------------------------------------
        logger.info(f"Validation for epoch number: {epoch}")

        accuracy = 0.0
        n = 0
        validStart = time.time()
        for image, label in val_loader:

            image = image.numpy().astype(dtype)
            image = jnp.moveaxis(image,(0,2,3,1),(0,1,2,3))
            label = label.numpy().astype(dtype)
            if image.shape[0] % num_devices != 0:
                continue

            # Reshape images from [num_devices * batch_size, height, width, img_channels]
            # to [num_devices, batch_size, height, width, img_channels].
            # The first dimension will be mapped across devices with jax.pmap.
            image = jnp.reshape(image, (num_devices, -1) + image.shape[1:])
            label = jnp.reshape(label, (num_devices, -1) + label.shape[1:])


            print("image before loss",image.shape)
            

            metrics = p_eval_step(state, {'image': image, 'label': label})
            accuracy += metrics['accuracy']
            n += 1
        accuracy /= n
        print('Validation accuracy:', jnp.mean(accuracy))
        accuracy = jnp.mean(accuracy).item()
        thisEpoch["valid_time"] = round(time.time() - validStart, 3)

        if accuracy > best_val_acc:
            best_val_acc = accuracy
            state = dataclasses.replace(state, **{'step': flax.jax_utils.replicate(step), 'epoch': flax.jax_utils.replicate(epoch)})
            save_checkpoint(state, jnp.mean(accuracy).item(), config.ckpt_dir)

        # update validation accuracy for this epoch
        thisEpoch["validation/accuracy"] = jnp.mean(accuracy).item()

        thisEpoch["epoch_time"] = round(time.time() - epochStart, 3)
        logger.info(f"Time performance summary for this EPOCH: {thisEpoch}\n")
        timeSummary.append(thisEpoch)
        if config.wandb:
            wandb.log(thisEpoch)

    totalEnd = time.time()
    logger.info(f"Full model training completed in {totalEnd - totalStart} seconds")
    fileName = os.path.join(config.work_dir, f"{config.name}.json")
    logger.info(f"Dumping time summary at: {fileName}")
    with open(fileName, "w") as jObj:
        json.dump(timeSummary, jObj, indent=2)


def main():
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument('--work_dir', default='/afs/crc.nd.edu/user/a/amaltar2/jax_tests/imagenette',
                        type=str, help='Directory for logging and checkpoints.')
    parser.add_argument('--data', default='/afs/crc.nd.edu/user/a/amaltar2/tensorflow_datasets/imagenette/320px-v2',
                        help='path to dataset (default: imagenet)')
    parser.add_argument('--name', type=str, default='test', help='Name of this experiment.')
    # Training
    parser.add_argument('--arch', type=str, default='vgg16', choices=['vgg16', 'vgg19'], help='Architecture.')
    parser.add_argument('--workers', default=4, type=int,
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--resume', action='store_true', help='Resume training from best checkpoint.')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--warmup_epochs', type=int, default=0, help='Number of warmup epochs with lower learning rate.') # no warmup epochs
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes.')
    parser.add_argument('--img_size', type=int, default=224, help='Image size.')
    parser.add_argument('--img_channels', type=int, default=3, help='Number of image channels.')
    parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision training.')
    parser.add_argument('--random_seed', type=int, default=0, help='Random seed.')
    # Logging
    parser.add_argument('--wandb', action='store_true', help='Log to Weights&bBiases.')
    parser.add_argument('--log_every', type=int, default=100, help='Log every log_every steps.')
    args = parser.parse_args()

    if jax.process_index() == 0:
        args.ckpt_dir = os.path.join(args.work_dir, args.name, 'checkpoints')
        if not os.path.exists(args.ckpt_dir):
            os.makedirs(args.ckpt_dir)

        if args.wandb:
            wandb.init(config=args,
                       dir=os.path.join(args.work_dir, args.name))

    train_and_evaluate(args)


if __name__ == '__main__':
    main()
