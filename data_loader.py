import os
import glob
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split

IMG_SIZE = (256, 256)
BATCH_SIZE = 75  # MesoNet paper value

def load_and_preprocess_image(path, label):
    """Reads an image from a file, decodes it into a dense tensor, and resizes it."""
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    # Normalize to [0, 1] range (paper standard)
    image = image / 255.0
    return image, label

def create_dataset_from_paths(paths, labels, shuffle=True, augment=False):
    """Creates a tf.data.Dataset from lists of paths and labels."""
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(paths), seed=42)

    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        # Augmentation as used in MesoNet-style implementations:
        # horizontal flip + mild brightness/contrast; keep it minimal to
        # match the paper's approach (the paper relied on architecture, not heavy aug).
        def augment_image(image, label):
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, max_delta=0.1)
            image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
            image = tf.clip_by_value(image, 0.0, 1.0)
            return image, label
        dataset = dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def load_data(base_dir='data',
              n_per_class=4900, seed=42):
    """
    Loads a perfectly balanced subset of the dataset.

    Randomly samples `n_per_class` images from each class (real=0, fake=1)
    using a fixed seed for reproducibility, then creates train/val/test splits.
    """
    random.seed(seed)

    all_real = glob.glob(os.path.join(base_dir, 'real', '*.png'))
    all_fake = glob.glob(os.path.join(base_dir, 'fake', '*.png'))

    print(f"Available real frames: {len(all_real)}")
    print(f"Available fake frames: {len(all_fake)}")

    # Sample exactly n_per_class from each class
    real_paths = random.sample(all_real, min(n_per_class, len(all_real)))
    fake_paths = random.sample(all_fake, min(n_per_class, len(all_fake)))

    print(f"Using {len(real_paths)} real + {len(fake_paths)} fake = {len(real_paths)+len(fake_paths)} total")

    paths  = real_paths + fake_paths
    labels = [0] * len(real_paths) + [1] * len(fake_paths)  # 0=Real, 1=Fake

    # 70% train / 15% val / 15% test  (stratified)
    X_tv, X_test, y_tv, y_test = train_test_split(
        paths, labels, test_size=0.15, stratify=labels, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.1765, stratify=y_tv, random_state=seed
    )

    print(f"Training samples:   {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples:       {len(X_test)}")

    train_dataset = create_dataset_from_paths(X_train, y_train, shuffle=True,  augment=True)
    val_dataset   = create_dataset_from_paths(X_val,   y_val,   shuffle=False)
    test_dataset  = create_dataset_from_paths(X_test,  y_test,  shuffle=False)

    return train_dataset, val_dataset, test_dataset


if __name__ == '__main__':
    train_ds, val_ds, test_ds = load_data()
    for images, labels in train_ds.take(1):
        print(f"Image batch shape: {images.shape}")
        print(f"Label batch shape: {labels.shape}")
        print("Successfully loaded datasets!")
