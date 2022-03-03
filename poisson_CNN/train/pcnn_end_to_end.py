import argparse, json, os
import datetime
import tensorflow as tf

# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['NCCL_DEBUG'] = 'INFO'
# os.environ.pop('TF_CONFIG', None)
# os.environ["TF_CONFIG"] = json.dumps({
#     'cluster': {
#         'worker': ["172.10.0.104:12345", "172.10.0.113:12345"]
#     },
#     'task': {'type': 'worker', 'index': 0}
# })
print(os.environ.get("TF_CONFIG"))


# physical_devices = tf.config.list_physical_devices('GPU')
# print(physical_devices)
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
# if len(physical_devices) > 0:
#     tf.config.experimental.set_memory_growth(physical_devices[0], False)

from .utils import load_model_checkpoint, choose_optimizer
from ..utils import convert_tf_object_names
from ..models import Homogeneous_Poisson_NN_Legacy, Dirichlet_BC_NN_Legacy_2, Poisson_CNN_Legacy
from ..losses import loss_wrapper
from ..dataset.generators import numerical_dataset_generator, reverse_poisson_dataset_generator


parser = argparse.ArgumentParser(description="Train the Homogeneous Poisson NN")
parser.add_argument("config", type=str, help="Path to the configuration json for training, model and dataset parameters")
parser.add_argument("--checkpoint_dir", type=str, help="Directory to save result checkpoints in", default=".")
parser.add_argument("--continue_from_checkpoint", type=str, help="Continue from this checkpoint file if provided", default=None)
parser.add_argument("--learning_rate", type=str, help="Overrides the learning rate with the provided value, or with the value from the json file if 'from_json' is the provided value", default = None)

args = parser.parse_args()

config = convert_tf_object_names(json.load(open(args.config)))
checkpoint_dir = args.checkpoint_dir

if 'precision' in config['training'].keys():
    tf.keras.backend.set_floatx(config['training']['precision'])

dataset = numerical_dataset_generator(randomize_boundary_smoothness = True, exclude_zero_boundaries = False, nonzero_boundaries = ['left','right','top','bottom'], rhses = 'random', return_boundaries=True, return_dx = True, return_rhs = True, **config['dataset'])

dist_strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.ReductionToOneDevice())#nccl causes errors on POWER9

# dist_strategy = tf.distribute.MultiWorkerMirroredStrategy()
with dist_strategy.scope():
    hpnn = Homogeneous_Poisson_NN_Legacy(**config['hpnn_model'])
    dbcnn = Dirichlet_BC_NN_Legacy_2(**config['dbcnn_model'])
    model = Poisson_CNN_Legacy(hpnn, dbcnn)
    inp, tar = dataset.__getitem__(0)
    out = model([x[:1] for x in inp])
    optimizer = choose_optimizer(config['training']['optimizer'])(**config['training']['optimizer_parameters'])
    loss = loss_wrapper(global_batch_size=config['dataset']['batch_size'], **config['training']['loss_parameters'])
    model.compile(loss=loss,optimizer=optimizer)


    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    cb = [
        tf.keras.callbacks.ModelCheckpoint(checkpoint_dir + '/chkpt-epoch-{epoch:02d}.mse-{mse:.4f}',save_weights_only=True,save_best_only=True,monitor = 'loss',verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(patience = 4,monitor='loss',min_lr=config['training']['min_learning_rate']),
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
        tf.keras.callbacks.TerminateOnNaN()
    ]

    # load_model_checkpoint(model, args.continue_from_checkpoint, model_config = config['model'], sample_model_input = inp)
    checkpoint_path = args.continue_from_checkpoint
    if checkpoint_path is not None:
        checkpoint_filename = tf.train.latest_checkpoint(checkpoint_path)
        print('Attempting to load checkpoint from ' + checkpoint_filename)
        model.load_weights(checkpoint_filename)


    if args.learning_rate is not None:
        model.optimizer.learning_rate = config['training']['optimizer_parameters']['learning_rate'] if args.learning_rate.lower() == 'from_json' else float(args.learning_rate)

    model.summary()
    model.run_eagerly = True
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    model.fit(dataset,epochs=config['training']['n_epochs'],callbacks = cb)