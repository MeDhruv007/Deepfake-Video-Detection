import os
import glob
import argparse
import tensorflow as tf
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, CSVLogger, LearningRateScheduler
)

from data_loader import load_data
from model import Meso4


def lr_schedule(epoch, lr):
    if epoch > 0 and epoch % 5 == 0:
        lr = max(lr * 0.5, 1e-6)
    return float(lr)


def train(epochs=50, learning_rate=1e-3, quick_test=False):
    print("Loading data...")
    train_ds, val_ds, test_ds = load_data()

    if quick_test:
        train_ds = train_ds.take(2)
        val_ds   = val_ds.take(1)
        epochs   = 2

    print("Building model...")
    model = Meso4(input_shape=(256, 256, 3))

    # Paper parameters: Adam, lr=1e-3
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks
    os.makedirs('weights', exist_ok=True)

    checkpoint = ModelCheckpoint(
        'weights/meso4_best.weights.h5',
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=True,
        mode='max',
        verbose=1
    )

    
    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=8,
        restore_best_weights=True,
        verbose=1
    )

    lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)
    csv_logger   = CSVLogger('training_log.csv', append=False)  

    print("Starting training...")
    print(f"  LR={learning_rate}, Batch=75, Max epochs={epochs}")
    print(f"  LR halved every 5 epochs (floor 1e-6)\n")

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[checkpoint, early_stop, lr_scheduler, csv_logger]
    )

    model.save_weights('weights/meso4_final.weights.h5')
    print("\nTraining finished! Weights saved in 'weights/' directory.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Meso4 — MesoNet paper params")
    parser.add_argument('--epochs', type=int,   default=50,   help='Max training epochs')
    parser.add_argument('--lr',     type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--quick-test', action='store_true',  help='2-batch smoke test')
    args = parser.parse_args()

    train(epochs=args.epochs, learning_rate=args.lr, quick_test=args.quick_test)


    