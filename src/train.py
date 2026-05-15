import argparse
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.dataset import create_dataset_from_csv
from src.model import build_resnet50
from src.utils import compute_class_weights, evaluate_binary


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def main(args):
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    train_ds, train_labels = create_dataset_from_csv(args.train_csv, img_root=args.img_root, img_size=(args.img_size, args.img_size), batch_size=args.batch_size, shuffle=True)
    val_ds, val_labels = create_dataset_from_csv(args.val_csv, img_root=args.img_root, img_size=(args.img_size, args.img_size), batch_size=args.batch_size, shuffle=False)
    test_ds, test_labels = create_dataset_from_csv(args.test_csv, img_root=args.img_root, img_size=(args.img_size, args.img_size), batch_size=args.batch_size, shuffle=False)

    class_weights = compute_class_weights(train_labels)

    model = build_resnet50(input_shape=(args.img_size, args.img_size, 3), base_trainable=args.unfreeze_base, dropout=args.dropout)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=args.lr), loss='binary_crossentropy', metrics=[keras.metrics.AUC(name='auc'), keras.metrics.Precision(name='precision'), keras.metrics.Recall(name='recall')])

    ckpt = keras.callbacks.ModelCheckpoint(os.path.join(args.output_dir, 'best_model.h5'), save_best_only=True, monitor='val_auc', mode='max')
    early = keras.callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=6, restore_best_weights=True)
    csv_logger = keras.callbacks.CSVLogger(os.path.join(args.output_dir, 'train_log.csv'))

    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, class_weight=class_weights, callbacks=[ckpt, early, csv_logger])

    # Save config and history
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Evaluate on test set
    y_prob = model.predict(test_ds)
    # flatten labels
    y_true = np.array(test_labels).astype(int)
    y_prob = y_prob.reshape(-1)
    metrics = evaluate_binary(y_true, y_prob)
    with open(os.path.join(args.output_dir, 'test_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print('Test metrics:', metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', required=True)
    parser.add_argument('--val_csv', required=True)
    parser.add_argument('--test_csv', required=True)
    parser.add_argument('--img_root', default='')
    parser.add_argument('--output_dir', default='runs')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--unfreeze_base', action='store_true', help='Unfreeze ResNet base for fine-tuning')

    args = parser.parse_args()
    main(args)
