import json
import tensorflow as tf
from configs.hyperparameters import *

def parse_config(args, verbose=False):
    # load basic configurations
    config = eval(args.config)
    
    # update configurations based on the command line arguments
    if args.lr_scheduler is not None:
        config["lr_scheduler"] = args.lr_scheduler
    if args.batch_size is not None:
        config['batch_size'] = int(args.batch_size)
    if args.epochs is not None:
        config['epochs'] = int(args.epochs)

    # add 'log_name' to the configuration
    config["log_name"] = f"{config['config_name']}_bs{config['batch_size']}_lrs{config['lr_scheduler']}"

    if verbose:
        print("Configuration:")
        print(json.dumps(config, indent=4))
    return config

def load_lr_scheduler(lr_scheduler, total_steps, total_epochs):
    # learning rate scheduler
    if lr_scheduler is not None:
        if lr_scheduler == 'cosine':
            decay_steps = total_steps * total_epochs # steps during the training
            initial_learning_rate = 0.001
            alpha = 1e-6
            lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate, decay_steps, alpha=alpha)
        if lr_scheduler == 'exponential':
            decay_steps = total_steps # typically the number of steps in one epoch
            initial_learning_rate = 0.001
            decay_rate = 0.9
            lr_decayed_fn = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate, decay_steps, decay_rate)
    else:
        lr_decayed_fn = 0.001
    return lr_decayed_fn