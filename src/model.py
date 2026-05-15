import tensorflow as tf

def build_resnet50(input_shape=(224,224,3), base_trainable=False, dropout=0.5):
    base = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=input_shape, pooling='avg')
    base.trainable = base_trainable

    x = base.output
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    out = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=base.input, outputs=out)
    return model
