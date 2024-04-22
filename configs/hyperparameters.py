# Options for hyperparameters:
# "lr_scheduler": [None, 'cosine', 'exponential']


mobilenet_small = {
    "config_name": "mobilenet_small",
    "epochs": 400,
    "img_height": 224,
    "img_width": 224,
    "lr_scheduler": 'cosine',
    "backbone_name": 'mobilenet_small',
    "load_weights": False,
    "trainable_backbone": True,
    "data_dir": "Sonar_pics_v3",
    "batch_size": 8,
    "checkpoint_dir": "ckpts",
    "results_dir": "results"
}

mobilenet_small_imagenet = {
    "config_name": "mobilenet_small_imagenet",
    "epochs": 400,
    "img_height": 224,
    "img_width": 224,
    "lr_scheduler": 'cosine',
    "backbone_name": 'mobilenet_small',
    "load_weights": True,
    "trainable_backbone": False,
    "data_dir": "Sonar_pics_v3",
    "batch_size": 8,
    "checkpoint_dir": "ckpts",
    "results_dir": "results"
}

mobilenet_large = {
    "config_name": "mobilenet_large",
    "epochs": 400,
    "img_height": 224,
    "img_width": 224,
    "lr_scheduler": 'cosine',
    "backbone_name": 'mobilenet_large',
    "load_weights": False,
    "trainable_backbone": True,
    "data_dir": "Sonar_pics_v3",
    "batch_size": 8,
    "checkpoint_dir": "ckpts",
    "results_dir": "results"
}

mobilenet_large_imagenet = {
    "config_name": "mobilenet_large_imagenet",
    "epochs": 400,
    "img_height": 224,
    "img_width": 224,
    "lr_scheduler": 'cosine',
    "backbone_name": 'mobilenet_large',
    "load_weights": True,
    "trainable_backbone": False,
    "data_dir": "Sonar_pics_v3",
    "batch_size": 8,
    "checkpoint_dir": "ckpts",
    "results_dir": "results"
}

efficientnet = {
    "config_name": "efficientnet",
    "epochs": 400,
    "img_height": 224,
    "img_width": 224,
    "lr_scheduler": 'cosine',
    "backbone_name": 'efficientnet',
    "load_weights": False,
    "trainable_backbone": True,
    "data_dir": "Sonar_pics_v3",
    "batch_size": 8,
    "checkpoint_dir": "ckpts",
    "results_dir": "results"
}

efficientnet_imagenet = {
    "config_name": "efficientnet_imagenet",
    "epochs": 400,
    "img_height": 224,
    "img_width": 224,
    "lr_scheduler": 'cosine',
    "backbone_name": 'efficientnet',
    "load_weights": True,
    "trainable_backbone": False,
    "data_dir": "Sonar_pics_v3",
    "batch_size": 8,
    "checkpoint_dir": "ckpts",
    "results_dir": "results"
}

lenet = {
    "config_name": "lenet",
    "epochs": 400,
    "img_height": 224,
    "img_width": 224,
    "lr_scheduler": 'cosine',
    "backbone_name": 'lenet',
    "load_weights": False,
    "trainable_backbone": True,
    "data_dir": "Sonar_pics_v3",
    "batch_size": 8,
    "checkpoint_dir": "ckpts",
    "results_dir": "results"
}

vgg = {
    "config_name": "vgg",
    "epochs": 400,
    "img_height": 224,
    "img_width": 224,
    "lr_scheduler": 'cosine',
    "backbone_name": 'vgg',
    "load_weights": False,
    "trainable_backbone": True,
    "data_dir": "Sonar_pics_v3",
    "batch_size": 8,
    "checkpoint_dir": "ckpts",
    "results_dir": "results"
}

vgg_imagenet = {
    "config_name": "vgg_imagenet",
    "epochs": 400,
    "img_height": 224,
    "img_width": 224,
    "lr_scheduler": 'cosine',
    "backbone_name": 'vgg',
    "load_weights": True,
    "trainable_backbone": False,
    "data_dir": "Sonar_pics_v3",
    "batch_size": 8,
    "checkpoint_dir": "ckpts",
    "results_dir": "results"
}