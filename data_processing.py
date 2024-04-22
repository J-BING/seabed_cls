import tensorflow as tf

IMG_SIZE = 224
AUTOTUNE = tf.data.AUTOTUNE

resize_and_rescale = tf.keras.Sequential([
       tf.keras.layers.Resizing(IMG_SIZE, IMG_SIZE),
       tf.keras.layers.Rescaling(1./255)
       ])

data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip(),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ])

def data_processing(ds, shuffle=False, augment=False):

  # Resize and rescale all datasets.
  ds = ds.map(lambda x, y: (resize_and_rescale(x), y), 
              num_parallel_calls=AUTOTUNE)

  if shuffle:
    ds = ds.shuffle(1000)

  # Use data augmentation only on the training set.
  if augment:
    ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), 
                num_parallel_calls=AUTOTUNE)

  # Use buffered prefetching on all datasets.
  return ds.prefetch(buffer_size=AUTOTUNE)