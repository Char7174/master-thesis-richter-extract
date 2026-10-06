import os
# Sets the environment variable for tensorflow messages to errors only.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow import keras
from tensorflow.keras import layers

from radesdnn import backgroundscript as bs
from radesdnn import callbacks_metrics as cm
from radesdnn import radardataset

# Set the subfolder for this model.
model_folder = '026_10_01_3DConv_LSTM_AG'


# ---------#
# I. Data  #
# ---------#

# Data Parameters
# Desciption: https://docs.python.org/3.13/library/stdtypes.html#dict
# Description: Parameters see class RadarDataset in module radardataset

data_param_dict = {
    'batch_size': 16,
    'class_distribution': {'all': [], 'train': [], 'val': [], 'test': []},
    'num_frames': 120,
    'patients': {'train': [4, 5, 6, 8],
                 'val': [2, 9],
                 'test': [3, 7]},
    'plus1d': True,
    'sample_counter': {'all': 0, 'train': 0, 'val': 0, 'test': 0},
    'segment_type_to_class': {'01': 0, '12': 1, '23': 1, '34': 2, '13': 1, '14': 1, '24': 1}, # like kida notebooks
    'variance_threshold': 0.0,
    'variance_threshold_class': False
}

# Create the RadarDataset

rd = radardataset.RadarDataset(num_frames = data_param_dict['num_frames'], 
                               patients = data_param_dict['patients'],
                               plus1d = data_param_dict['plus1d'],
                               segment_type_to_class = data_param_dict['segment_type_to_class'],
                               variance_threshold = data_param_dict['variance_threshold'],
                               variance_threshold_class = data_param_dict['variance_threshold_class'])


#------------#
# II. Model  #
#------------#

# Model Parameters

model_param_dict = {
    'num_blocks': 3,
    'num_filters': [64, 64, 64]
}

# Model Definition

def create_model(input_shape, num_classes):
    """
    3D-Conv for feature extraction
    LSTM for time series
    Dense for classification
    """
    
    inputs = keras.Input(shape=input_shape)
    x = inputs
    
    # feature extraction
    for i in range(model_param_dict['num_blocks']):
        x = layers.Conv3D(model_param_dict['num_filters'][i], 
                          (3, 3, 3),
                          activation=None, 
                          padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation='relu')(x)
        if x.shape[-2] > 24:
            x = layers.MaxPooling3D((2, 1, 2))(x)
        X = layers.Dropout(0.2)(x)
        
    x = layers.TimeDistributed(
        layers.Flatten())(x)
    
    # regarding time series
    # x = layers.LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(x)
    x = layers.LSTM(32, dropout=0.2, recurrent_dropout=0.2)(x)
    
    # classification
    x = layers.Dense(128, activation='relu')(x)
    X = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


#----------------#
# III. Training  #
#----------------#

# Training Parameters

training_param_dict = {
    'augmentation': True,
    'callbacks': [cm.checkpoint(model_folder)],
    'epochs': 30, 
    'learning_rate': 2e-5,
    'loss': 'categorical_crossentropy',
    'metrics': ['categorical_accuracy', cm.f2_score],
    'optimizer': keras.optimizers.Adam()
}


# Start the Whole Process

bs.start(
    create_model=create_model,
    data_param_dict=data_param_dict,
    gen_name='standardization_on_parts',
    model_folder=model_folder,
    model_param_dict=model_param_dict,
    rd=rd,
    training_param_dict=training_param_dict,
    tuner=False
)