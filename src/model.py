from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Add,
    MaxPooling2D, AveragePooling2D, Flatten,
    Dense, Dropout
)
from src.cbam import cbam_block  # Adjust the import based on your folder structure

def build_cnn_model(dropout_rate=0.3, input_shape=(224, 224, 4)):
    input_tensor = Input(shape=input_shape)
    
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Add()([
        MaxPooling2D(pool_size=(2, 2))(x),
        AveragePooling2D(pool_size=(2, 2))(x)
    ])
    x = cbam_block(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([
        MaxPooling2D(pool_size=(2, 2))(x),
        AveragePooling2D(pool_size=(2, 2))(x)
    ])
    x = cbam_block(x)

    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=input_tensor, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model
