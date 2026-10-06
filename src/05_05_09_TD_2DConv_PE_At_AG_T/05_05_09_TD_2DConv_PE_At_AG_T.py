import os
# Sets the environment variable for tensorflow messages to errors only.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

from radesdnn import backgroundscript as bs
from radesdnn import callbacks_metrics as cm
from radesdnn import radardataset

# Set the subfolder for this model.
model_folder = '05_05_09_TD_2DConv_PE_At_AG_T'


# ---------#
# I. Data  #
# ---------#

# Data Parameters
# Desciption: https://docs.python.org/3.13/library/stdtypes.html#dict
# Description: Parameters see class RadarDataset in module radardataset

data_param_dict = {
    'batch_size': 32,
    'class_distribution': {'all': [], 'train': [], 'val': [], 'test': []},
    'num_frames': 64,
    'patients': {'train': [9],
                 'val': [6],
                 'test': [3]},
    'plus1d': True,
    'sample_counter': {'all': 0, 'train': 0, 'val': 0, 'test': 0},
    'segment_type_to_class': {'01': 0, '12': 1, '23': 2, '34': 3, '13': -1, 
                              '14': -1, '24': -1},
    'seizure_types': ['konvulsiv'],
    'variance_threshold': 7.0e-20,
    'variance_threshold_class': True
}

# Create the RadarDataset

rd = radardataset.RadarDataset(
            num_frames = data_param_dict['num_frames'],
            patients = data_param_dict['patients'],
            plus1d = data_param_dict['plus1d'],
            segment_type_to_class = data_param_dict['segment_type_to_class'],
            seizure_types = data_param_dict['seizure_types'],
            variance_threshold = data_param_dict['variance_threshold'],
            variance_threshold_class = data_param_dict['variance_threshold_class'])


#------------#
# II. Model  #
#------------#

# Model Parameters

model_param_dict = {
    'dropout': [0.1, 0.3, 0.5],
    'filters': [64, 32, 16],
    'key_dim': [64, 128, 256],
    'max_blocks': 4,
    'max_dense': 2,
    'num_heads': [8],
    'units': [64, 32, 8],
}


# Based on: https://keras.io/examples/timeseries/timeseries_classification_transformer/
def create_model(hp, input_shape, num_classes):
    """
    2D-CNN for feature extraction
    MultiHeadAttention for attention
    Dense for classification
    """

    dim = 128
    
    blocks = hp.Int('blocks', 2, model_param_dict['max_blocks'])
    dense = hp.Int('dense', 1, model_param_dict['max_dense'])
    dropout = hp.Choice('dropout', model_param_dict['dropout'])
    filters = hp.Choice('filters', model_param_dict['filters'])
    key_dim = hp.Choice('key_dim', model_param_dict['key_dim'])
    units = hp.Choice('units', model_param_dict['units'])
    num_heads = hp.Choice('num_heads', model_param_dict['num_heads'])

    inputs = keras.Input(shape=input_shape)
    x = inputs
    
    x = layers.TimeDistributed(
        layers.Conv2D(32, 
                      (3, 3),
                      activation='linear', 
                      padding='same'))(x)
    
    x = layers.TimeDistributed(
        layers.Conv2D(64, 
                      (3, 3),
                      activation='linear', 
                      padding='same'))(x)

    x = layers.TimeDistributed(
        layers.GlobalAveragePooling2D())(x)
    x = layers.Dense(dim, activation='relu')(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.1)(x)
    
    pe = layers.Embedding(input_dim=input_shape[0], output_dim=dim)
    pos = tf.range(start=0, limit=input_shape[0], delta=1)
    pos_enc = pe(pos)
    x = x + pos_enc
    
    # attention
    att = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=dim // num_heads,
            dropout=dropout,
            name='mha')(x, x)
    x = layers.Add()([x, att])
    x = layers.LayerNormalization()(x)
    
    x = layers.GlobalAveragePooling1D()(x)
        
    # classification
    for i in range(dense):
        x = layers.Dense(units, activation='sigmoid')(x)
        x = layers.Dropout(dropout)(x)
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
    'epochs': 100, 
    'learning_rate': 1e-4,
    'loss': 'categorical_crossentropy',
    'metrics': ['categorical_accuracy', cm.f2_score],
    'optimizer': keras.optimizers.Adam(),
    'samples_per_class': 0,
    'tuner': True
}


# Start the Whole Process

bs.start(
    create_model=create_model,
    data_param_dict=data_param_dict,
    gen_name='no_outliers_replaced',
    model_folder=model_folder,
    model_param_dict=model_param_dict,
    rd=rd,
    training_param_dict=training_param_dict,
)