import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import json
import tensorflow as tf
import horovod.tensorflow.keras as hvd
import argparse

import datetime

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
hvd.init()
print(">>>> hvd.rank:", hvd.rank(), "hvd.size:", hvd.size())

# Horovod: pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# if args.use_fp16:
# tf.keras.mixed_precision.set_global_policy('mixed_float16')

if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

from poisson_CNN.train.utils import load_model_checkpoint, choose_optimizer
from poisson_CNN.utils import convert_tf_object_names
from poisson_CNN.models import Homogeneous_Poisson_NN_Legacy
from poisson_CNN.losses import loss_wrapper
from poisson_CNN.dataset.generators import numerical_dataset_generator, reverse_poisson_dataset_generator

parser = argparse.ArgumentParser(description="Train the Homogeneous Poisson NN")
parser.add_argument("config", type=str,
                    help="Path to the configuration json for training, model and dataset parameters")
parser.add_argument("--checkpoint_dir", type=str, help="Directory to save result checkpoints in", default=".")
parser.add_argument("--continue_from_checkpoint", type=str, help="Continue from this checkpoint file if provided",
                    default=None)
parser.add_argument("--dataset_type", type=lambda x: str(x).lower(),
                    help="Method of dataset generation. Options are 'numerical' or 'analytical'.", default="analytical")
parser.add_argument("--learning_rate", type=str,
                    help="Overrides the learning rate with the provided value, or with the value from the json file if 'from_json' is the provided value",
                    default=None)

args = parser.parse_args()
# Horovod: write logs on worker 0.
verbose = 1 if hvd.rank() == 0 else 0
# verbose = 1
recognized_dataset_types = ['numerical', 'analytical']
if args.dataset_type not in recognized_dataset_types:
    raise (ValueError(
        'Invalid dataset type. Received: ' + args.dataset_type + ' | Recognized values: ' + recognized_dataset_types))

config = convert_tf_object_names(json.load(open(args.config)))
checkpoint_dir = args.checkpoint_dir

if 'precision' in config['training'].keys():
    tf.keras.backend.set_floatx(config['training']['precision'])

if args.dataset_type == 'numerical':
    dataset = numerical_dataset_generator(**config['dataset'])
elif args.dataset_type == 'analytical':
    dataset = reverse_poisson_dataset_generator(**config['dataset'])

model = Homogeneous_Poisson_NN_Legacy(**config['model'])
optimizer = choose_optimizer(config['training']['optimizer'])(**config['training']['optimizer_parameters'])
loss = loss_wrapper(global_batch_size=config['dataset']['batch_size'], **config['training']['loss_parameters'])

inp, tar = dataset.__getitem__(0)
out = model([inp[0][:1], inp[1][:1, :1]])
# optimizer = hvd.DistributedOptimizer(optimizer)
model.compile(loss=loss, optimizer=optimizer)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
cb = [
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),
    ## step 5: Broadcasting the variables for same weights initialization on all workers or for continuing from cheakpoint.
    hvd.callbacks.MetricAverageCallback(),
    hvd.callbacks.LearningRateWarmupCallback(initial_lr=config['training']['optimizer_parameters']['learning_rate'],
                                             warmup_epochs=3, verbose=verbose),
    tf.keras.callbacks.ReduceLROnPlateau(patience=4, monitor='loss', min_lr=config['training']['min_learning_rate']),
    tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
    tf.keras.callbacks.TerminateOnNaN()
]

# Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
if hvd.rank() == 0:
    cb.append(tf.keras.callbacks.ModelCheckpoint(checkpoint_dir + '/chkpt-epoch-{epoch:02d}.mse-{mse:.4f}',
                                                 save_weights_only=True, save_best_only=True, monitor='loss',
                                                 verbose=verbose))

# cb = [
#     tf.keras.callbacks.ModelCheckpoint(checkpoint_dir + '/chkpt-epoch-{epoch:02d}.mse-{mse:.4f}', save_weights_only=True, save_best_only=True, monitor='mse', verbose=1),
#     tf.keras.callbacks.ReduceLROnPlateau(patience = 4,monitor='loss',min_lr=config['training']['min_learning_rate']),
#     tf.keras.callbacks.TerminateOnNaN()
# ]

load_model_checkpoint(model, args.continue_from_checkpoint, model_config=config['model'], sample_model_input=inp)

if args.learning_rate is not None:
    model.optimizer.learning_rate = config['training']['optimizer_parameters'][
        'learning_rate'] if args.learning_rate.lower() == 'from_json' else float(args.learning_rate)

model.optimizer.learning_rate = model.optimizer.learning_rate * hvd.size()  ## step 6: proportionate increase in learning rate.

if hvd.size() == 0:
    model.summary()

# model.summary()
model.run_eagerly = True
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
model.fit(dataset, epochs=config['training']['n_epochs'], callbacks=cb, verbose=verbose, initial_epoch=7)
