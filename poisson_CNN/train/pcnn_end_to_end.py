import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import json
import tensorflow as tf
import horovod.tensorflow.keras as hvd
import argparse

import datetime
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
hvd.init()
# print(">>>> hvd.rank:", hvd.rank(), "hvd.size:", hvd.size())

# Horovod: pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

#if args.use_fp16:
# tf.keras.mixed_precision.set_global_policy('mixed_float16')

if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

from poisson_CNN.train.utils import load_model_checkpoint, choose_optimizer
from poisson_CNN.utils import convert_tf_object_names
from poisson_CNN.models import Homogeneous_Poisson_NN_Legacy, Dirichlet_BC_NN_Legacy_2, Poisson_CNN_Legacy
from poisson_CNN.losses import loss_wrapper
from poisson_CNN.dataset.generators import numerical_dataset_generator, reverse_poisson_dataset_generator

parser = argparse.ArgumentParser(description="Train the Homogeneous Poisson NN")
parser.add_argument("config", type=str, help="Path to the configuration json for training, model and dataset parameters")
parser.add_argument("--checkpoint_dir", type=str, help="Directory to save result checkpoints in", default=".")
parser.add_argument("--continue_from_checkpoint", type=str, help="Continue from this checkpoint file if provided", default=None)
parser.add_argument("--learning_rate", type=str, help="Overrides the learning rate with the provided value, or with the value from the json file if 'from_json' is the provided value", default = None)
args = parser.parse_args()

# Horovod: write logs on worker 0.
verbose = 1 if hvd.rank() == 0 else 0

config = convert_tf_object_names(json.load(open(args.config)))
checkpoint_dir = args.checkpoint_dir

if 'precision' in config['training'].keys():
    tf.keras.backend.set_floatx(config['training']['precision'])

dataset = numerical_dataset_generator(randomize_boundary_smoothness = True, exclude_zero_boundaries = False, nonzero_boundaries = ['left','right','top','bottom'], rhses = 'random', return_boundaries=True, return_dx = True, return_rhs = True, **config['dataset'])

hpnn = Homogeneous_Poisson_NN_Legacy(**config['hpnn_model'])
dbcnn = Dirichlet_BC_NN_Legacy_2(**config['dbcnn_model'])
model = Poisson_CNN_Legacy(hpnn, dbcnn)
inp, tar = dataset.__getitem__(0)
out = model([x[:1] for x in inp])

optimizer = choose_optimizer(config['training']['optimizer'])(**config['training']['optimizer_parameters'])

# optimizer = hvd.DistributedOptimizer(optimizer)         ## step 4: binding the gradient for avg gradient calculation

loss = loss_wrapper(global_batch_size=config['dataset']['batch_size'], **config['training']['loss_parameters'])
model.compile(loss=loss,optimizer=optimizer) #, experimental_run_tf_function=False)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
cb = [
    hvd.callbacks.BroadcastGlobalVariablesCallback(0), ## step 5: Broadcasting the variables for same weights initialization on all workers or for continuing from cheakpoint.
    hvd.callbacks.MetricAverageCallback(),
    hvd.callbacks.LearningRateWarmupCallback(initial_lr=config['training']['optimizer_parameters']['learning_rate'], warmup_epochs=3, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(patience = 4,monitor='loss',min_lr=config['training']['min_learning_rate']),
    tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
    tf.keras.callbacks.TerminateOnNaN()
]

# Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
if hvd.rank() == 0:
    cb.append(tf.keras.callbacks.ModelCheckpoint(checkpoint_dir + '/chkpt-epoch-{epoch:02d}.mse-{mse:.4f}',save_weights_only=True,save_best_only=True,monitor = 'loss',verbose=1))


# load_model_checkpoint(model, args.continue_from_checkpoint, model_config = config['model'], sample_model_input = inp)
checkpoint_path = args.continue_from_checkpoint
if checkpoint_path is not None:
    checkpoint_filename = tf.train.latest_checkpoint(checkpoint_path)
    print('Attempting to load checkpoint from ' + checkpoint_filename)
    model.load_weights(checkpoint_filename)


if args.learning_rate is not None:
    model.optimizer.learning_rate = config['training']['optimizer_parameters']['learning_rate'] if args.learning_rate.lower() == 'from_json' else float(args.learning_rate)

model.optimizer.learning_rate = model.optimizer.learning_rate * hvd.size()  ## step 6: proportionate increase in learning rate.


if hvd.size() == 0:
    model.summary()

model.run_eagerly = True

# model.train_on_batch(dataset,epochs=config['training']['n_epochs'],callbacks = cb)
model.fit(dataset, epochs=config['training']['n_epochs'], callbacks=cb, verbose=verbose)

# print("Done")