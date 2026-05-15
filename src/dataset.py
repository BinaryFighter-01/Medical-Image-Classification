import tensorflow as tf
import pandas as pd
import os

def _load_and_preprocess(path, label, img_size=(224,224)):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, img_size)
    image = tf.keras.applications.resnet50.preprocess_input(image)
    return image, tf.cast(label, tf.float32)

def create_dataset_from_csv(csv_path, img_root="", img_size=(224,224), batch_size=32, shuffle=True):
    df = pd.read_csv(csv_path)
    if 'image_path' not in df.columns or 'label' not in df.columns:
        raise ValueError('CSV must contain image_path and label columns')

    paths = df['image_path'].apply(lambda p: os.path.join(img_root, p) if img_root and not os.path.isabs(p) else p).tolist()
    labels = df['label'].tolist()

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths))
    ds = ds.map(lambda p, l: _load_and_preprocess(p, l, img_size), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, labels
