from tensorflow.keras import layers
from tensorflow.keras import Model

def double_conv_block(x, num_filters):
    # Conv2D + ReLU activation
    x = layers.Conv2D(num_filters, 3, padding='same', activation='relu')(x)

    # Conv2D + ReLU activation
    x = layers.Conv2D(num_filters, 3, padding='same', activation='relu')(x)

    return x

def downsampling_block(x, num_filters):
    # Double convolution block
    skip_connection = double_conv_block(x, num_filters)

    # MaxPooling
    p = layers.MaxPooling2D()(skip_connection)

    # Dropout 0.3
    p = layers.Dropout(rate=0.3)(p)

    return skip_connection, p

def upsampling_block(x, skip_connection, num_filters):
    # Upsample
    x = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(x)

    # Concatenate
    x = layers.concatenate([x, skip_connection], axis=-1)

    # Double convolution block
    x = double_conv_block(x, num_filters)

    return x

def U_NET(width, height, channels, n_classes):
    inputs = layers.Input(shape=(width, height, channels))

    # Encoder
    # downsample 1
    f1, d1 = downsampling_block(inputs, 64)
    
    # downsample 2
    f2, d2 = downsampling_block(d1, 128)

    # downsample 3
    f3, d3 = downsampling_block(d2, 256)

    # downsample 4
    f4, d4 = downsampling_block(d3, 512)

    # bottleneck
    bottleneck = double_conv_block(d4, 1024)

    # Decoder
    # upsample 1
    u1 = upsampling_block(bottleneck, f4, 512)

    # upsample 2
    u2 = upsampling_block(u1, f3, 256)

    # upsample 3
    u3 = upsampling_block(u2, f2, 128)

    # upsample 4
    u4 = upsampling_block(u3, f1, 64)

    # outputs
    outputs = layers.Conv2D(n_classes, kernel_size=(1,1), 
                            padding='same', activation = "softmax")(u4)

    unet_model = Model(inputs=inputs, outputs=outputs, name="U-Net")

    return unet_model

if __name__ == "__main__":
    model = U_NET(128, 128, 3, 3)
    model.summary()