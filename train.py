import argparse
import os
import tensorflow as tf
import keras
import datetime
from data_processing import data_processing
from models import backbone_model, create_model
import json
from configs.config import parse_config, load_lr_scheduler

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='efficientnet', help = 'Enter the config, such as mobilenet_small, mobilenet_small_imagenet. Details refer to configs/hyperparameter.py.')
parser.add_argument('--lr_scheduler', default='cosine', help = 'Enter the learning rate scheduler, such as cosine, exponential.')
parser.add_argument('--batch_size', default=8, help = 'Enter the batch size.')
parser.add_argument('--epochs', default=400, help = 'Enter the number of epochs.')
args = parser.parse_args()

config = parse_config(args, verbose=True)

epochs = config["epochs"]
img_height = config["img_height"]
img_width = config["img_width"]
lr_scheduler = config["lr_scheduler"]
backbone_name = config["backbone_name"]
load_weights = config["load_weights"]
trainable_backbone = config["trainable_backbone"]
data_dir = config["data_dir"]
batch_size = config["batch_size"]
checkpoint_dir = config["checkpoint_dir"]
resutls_dir = config["results_dir"]
log_name = config["log_name"]

if os.path.exists(checkpoint_dir) is False:
    os.makedirs(checkpoint_dir)
if os.path.exists(resutls_dir) is False:
    os.makedirs(resutls_dir)

######################################################
timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

# Load the dataset
train_ds = tf.keras.utils.image_dataset_from_directory(directory=data_dir,labels="inferred",
                                                       label_mode="int",batch_size=batch_size,image_size=(img_height,img_width),
                                                       shuffle=True,seed=123,validation_split=0.2,
                                                       subset="training")
val_ds = tf.keras.utils.image_dataset_from_directory(directory=data_dir,labels="inferred", label_mode="int",
                                                        batch_size=batch_size,image_size=(img_height,img_width),
                                                        shuffle=True,seed=123,validation_split=0.2,
                                                        subset="validation")
num_class = len(train_ds.class_names)
num_batches = train_ds.cardinality().numpy()

# process the data
train_ds_augmented = data_processing(train_ds, shuffle=True, augment=True)
val_ds = data_processing(val_ds)

# Create the model
model = create_model(backbone_name, num_class, load_weights=load_weights, trainable_backbone=trainable_backbone)
# print the model information
print(model.summary(show_trainable=True))

# Load the learning rate scheduler
lr_decayed_fn = load_lr_scheduler(lr_scheduler, total_steps=num_batches, total_epochs=epochs)
optimizer = keras.optimizers.Adam(learning_rate=lr_decayed_fn)
# metric1 = keras.metrics.SparseCategoricalAccuracy()

# Compile the model
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

# Create a callback that saves the model's weights
checkpoint_path = os.path.join(checkpoint_dir, f"{log_name}_{timestamp}/cp.weights.h5")
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                save_weights_only=True,
                                                save_best_only=True,
                                                verbose=1)

# Train and evaluate the model
start_time = datetime.datetime.now()
history = model.fit(train_ds_augmented, epochs=epochs, validation_data=val_ds, callbacks=[cp_callback])
end_time = datetime.datetime.now()
total_training_time = str(end_time - start_time)

# Save training history to results directory
history_file = os.path.join(resutls_dir, f"results_{log_name}_{timestamp}.json")
history_dict = history.history
history_dict["total_training_time"] = total_training_time
with open(history_file, "w") as f:
    json.dump(history_dict, f)