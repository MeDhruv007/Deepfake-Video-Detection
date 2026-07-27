import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout, LeakyReLU

def Meso4(input_shape=(256, 256, 3)):
    """
    Builds the Meso-4 model architecture as described in the DeepFake detection literature.
    It comprises 4 blocks of convolutional layers followed by a fully connected network.
    """
    x = Input(shape=input_shape)
    
    # Block 1
    x1 = Conv2D(8, (3, 3), padding='same', activation='relu')(x)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)
    
    # Block 2
    x2 = Conv2D(8, (5, 5), padding='same', activation='relu')(x1)
    x2 = BatchNormalization()(x2)
    x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)
    
    # Block 3
    x3 = Conv2D(16, (5, 5), padding='same', activation='relu')(x2)
    x3 = BatchNormalization()(x3)
    x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)
    
    # Block 4
    x4 = Conv2D(16, (5, 5), padding='same', activation='relu')(x3)
    x4 = BatchNormalization()(x4)
    x4 = MaxPooling2D(pool_size=(4, 4), padding='same')(x4)
    
    # Fully Connected Network
    y = Flatten()(x4)
    y = Dropout(0.5)(y)
    y = Dense(16)(y)
    y = LeakyReLU(negative_slope=0.1)(y)
    y = Dropout(0.5)(y)
    
    # Output layer for Binary Classification
    y = Dense(1, activation='sigmoid')(y)
    
    model = Model(inputs=x, outputs=y, name="Meso4")
    return model

if __name__ == '__main__':
    model = Meso4()
    model.summary()
