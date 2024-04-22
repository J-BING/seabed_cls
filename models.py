import keras_cv
import tensorflow as tf

def create_lenet(input_shape=(224, 224, 1)):
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding="same"),
            tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
            tf.keras.layers.Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'),
        ]
    
    )
    model(tf.keras.layers.Input(shape=input_shape))
    return model

def backbone_model(name='mobilenet_small', load_weights=False, trainable=False):

    if name == 'mobilenet_small':
        backbone = keras_cv.models.MobileNetV3Backbone.from_preset(
            'mobilenet_v3_small_imagenet',
            load_weights=load_weights,)
    elif name == 'mobilenet_large':
        backbone = keras_cv.models.MobileNetV3Backbone.from_preset(
            'mobilenet_v3_large_imagenet',
            load_weights=load_weights,)
    elif name == 'resnet18':
        backbone = keras_cv.models.ResNetV2Backbone.from_preset(
            'resnet18_v2',
            load_weights=None,)
    elif name == 'resnet34':
        backbone = keras_cv.models.ResNetV2Backbone.from_preset(
            'resnet34_v2',
            load_weights=None,)
    elif name == 'efficientnet':
        backbone = keras_cv.models.EfficientNetV2Backbone.from_preset(
            'efficientnetv2_b0_imagenet',
            load_weights=load_weights,)
    elif name == 'vgg':
        if load_weights:
            weights = 'imagenet'
        else:
            weights = None
        backbone = tf.keras.applications.vgg16.VGG16(include_top=False, weights=weights)
    elif name == 'lenet':
        backbone = create_lenet(input_shape=(224, 224, 3))
    else:
        raise ValueError(f"Unknown backbone: {name}")
    
    backbone.trainable = trainable
    return backbone

def create_model(backbone_name, num_class, load_weights=False, trainable_backbone=True):
    backbone = backbone_model(name=backbone_name, load_weights=load_weights, trainable=trainable_backbone)
    pooling_layer = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")
    # add dropout layer
    inputs = backbone.input
    x = backbone(inputs)
    x = pooling_layer(x)
    classifier = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_class, activation='softmax')
    ], name='classifier')
    outputs = classifier(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model