import argparse
import glob
import importlib
import os
import numpy as np
import shutil
import torch

from data import load_data, get_subject_files
from model import Model
from minibatching import (iterate_minibatches,
                          iterate_batch_seq_minibatches,
                          iterate_batch_multiple_seq_minibatches)
from utils import (get_balance_class_oversample,
                   print_n_samples_each_class,
                   compute_portion_each_class,
                   save_seq_ids,
                   load_seq_ids)
from logger import get_logger

import logging
# tf.get_logger().setLevel(logging.ERROR)


def train(
    args,
    config_file,
    fold_idx,
    output_dir,
    log_file,
    restart=False,
    random_seed=42,
):
    spec = importlib.util.spec_from_file_location("*", config_file)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    config = config.train



    # Create output directory for the specified fold_idx
    output_dir = os.path.join(output_dir, str(fold_idx))
    if restart:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
    else:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    # Create logger
    logger = get_logger(log_file, level="info")

    subject_files = glob.glob(os.path.join(config["data_dir"], "*.npz"))

    # Load subject IDs
    fname = "{}.txt".format(config["dataset"])
    seq_sids = load_seq_ids(fname)
    logger.info("Load generated SIDs from {}".format(fname))
    logger.info("SIDs ({}): {}".format(len(seq_sids), seq_sids))

    # Split training and test sets
    fold_pids = np.array_split(seq_sids, config["n_folds"])

    test_sids = fold_pids[fold_idx]
    train_sids = np.setdiff1d(seq_sids, test_sids)

    # Further split training set as validation set (10%)
    n_valids = round(len(train_sids) * 0.10)

    # Set random seed to control the randomness
    np.random.seed(random_seed)
    valid_sids = np.random.choice(train_sids, size=n_valids, replace=False)
    train_sids = np.setdiff1d(train_sids, valid_sids)

    logger.info("Train SIDs: ({}) {}".format(len(train_sids), train_sids))
    logger.info("Valid SIDs: ({}) {}".format(len(valid_sids), valid_sids))
    logger.info("Test SIDs: ({}) {}".format(len(test_sids), test_sids))

    # Get corresponding files
    train_files = []
    for sid in train_sids:
        train_files.append(get_subject_files(
            dataset=config["dataset"],
            files=subject_files,
            sid=sid,
        ))
    train_files = np.hstack(train_files)
    train_x, train_y, _ = load_data(train_files)

    valid_files = []
    for sid in valid_sids:
        valid_files.append(get_subject_files(
            dataset=config["dataset"],
            files=subject_files,
            sid=sid,
        ))
    valid_files = np.hstack(valid_files)
    valid_x, valid_y, _ = load_data(valid_files)

    test_files = []
    for sid in test_sids:
        test_files.append(get_subject_files(
            dataset=config["dataset"],
            files=subject_files,
            sid=sid,
        ))
    test_files = np.hstack(test_files)
    test_x, test_y, _ = load_data(test_files)

    # Print training, validation and test sets
    logger.info("Training set (n_night_sleeps={})".format(len(train_y)))
    for _x in train_x: logger.info(_x.shape)
    print_n_samples_each_class(np.hstack(train_y))
    logger.info("Validation set (n_night_sleeps={})".format(len(valid_y)))
    for _x in valid_x: logger.info(_x.shape)
    print_n_samples_each_class(np.hstack(valid_y))
    logger.info("Test set (n_night_sleeps={})".format(len(test_y)))
    for _x in test_x: logger.info(_x.shape)
    print_n_samples_each_class(np.hstack(test_y))

    # Force to use 1.5 only for N1
    if config.get('weighted_cross_ent') is None:
        config['weighted_cross_ent'] = False
        logger.info(f'  Weighted cross entropy: Not specified --> default: {config["weighted_cross_ent"]}')
    else:
        logger.info(f'  Weighted cross entropy: {config["weighted_cross_ent"]}')
    if config['weighted_cross_ent']:
        config["class_weights"] = np.asarray([1., 1.5, 1., 1., 1.], dtype=np.float32)
    else:
        config["class_weights"] = np.asarray([1., 1., 1., 1., 1.], dtype=np.float32)
    logger.info(f'  Weighted cross entropy: {config["class_weights"]}')

    # Create a model
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    logger.info(f'using device {args.gpu}')
    model = Model(
        config=config,
        output_dir=output_dir,
        use_rnn=True,
        testing=False,
        use_best=False,
        device=device
    )

    # Data Augmentation Details
    logger.info('Data Augmentation')
    logger.info(f'  Sequence: {config["augment_seq"]}')
    logger.info(f'  Signal full: {config["augment_signal_full"]}')

    # Train using epoch scheme
    best_acc = -1
    best_mf1 = -1
    update_epoch = -1
    for epoch in range(model.get_current_epoch(), config["n_epochs"]):
        # Create minibatches for training
        shuffle_idx = np.random.permutation(np.arange(len(train_x)))  # shuffle every epoch is good for generalization
        # Create augmented data
        percent = 0.1
        aug_train_x = np.copy(train_x)
        aug_train_y = np.copy(train_y)
        for i in range(len(aug_train_x)):
            # Shift signals horizontally
            offset = np.random.uniform(-percent, percent) * aug_train_x[i].shape[1]
            roll_x = np.roll(aug_train_x[i], int(offset))
            if offset < 0:
                aug_train_x[i] = roll_x[:-1]
                aug_train_y[i] = aug_train_y[i][:-1]
            if offset > 0:
                aug_train_x[i] = roll_x[1:]
                aug_train_y[i] = aug_train_y[i][1:]
            roll_x = None
            assert len(aug_train_x[i]) == len(aug_train_y[i])
        aug_minibatch_fn = iterate_batch_multiple_seq_minibatches(
            aug_train_x,
            aug_train_y,
            batch_size=config["batch_size"],
            seq_length=config["seq_length"],
            shuffle_idx=shuffle_idx,
            augment_seq=config['augment_seq'],
        )
        # Train, one epoch,
        train_outs = model.train_with_dataloader(aug_minibatch_fn)  # 只使用增强后的数据进行训练， 每个epoch进行一次数据增强
        # todo add a bank, and clustering, using protoNCE

        # Create minibatches for validation

        valid_minibatch_fn = iterate_batch_multiple_seq_minibatches(
            valid_x,
            valid_y,
            batch_size=config["batch_size"],
            seq_length=config["seq_length"],
            shuffle_idx=None,
            augment_seq=False,
        )
        valid_outs = model.evaluate_with_dataloader(valid_minibatch_fn)

        # Create minibatches for testing
        test_minibatch_fn = iterate_batch_multiple_seq_minibatches(
            test_x,
            test_y,
            batch_size=config["batch_size"],
            seq_length=config["seq_length"],
            shuffle_idx=None,
            augment_seq=False,
        )
        test_outs = model.evaluate_with_dataloader(test_minibatch_fn)


        logger.info("[e{}/{} s{}] TR (n={}) l={:.4f} a={:.1f} f1={:.1f} ({:.1f}s)| "
                    "VA (n={}) l={:.4f} a={:.1f}, f1={:.1f} ({:.1f}s) | "
                    "TE (n={}) l={:.4f} a={:.1f}, f1={:.1f} ({:.1f}s)".format(
            epoch+1,
            config["n_epochs"],
            train_outs["global_step"],
            len(train_outs["train/trues"]),
            train_outs["train/loss"],
            train_outs["train/accuracy"] * 100,
            train_outs["train/f1_score"] * 100,
            train_outs["train/duration"],

            len(valid_outs["test/trues"]),
            valid_outs["test/loss"],
            valid_outs["test/accuracy"] * 100,
            valid_outs["test/f1_score"] * 100,
            valid_outs["test/duration"],

            len(test_outs["test/trues"]),
            test_outs["test/loss"],
            test_outs["test/accuracy"] * 100,
            test_outs["test/f1_score"] * 100,
            test_outs["test/duration"],
            )
        )
        # Check best model
        # if best_acc < valid_outs["test/accuracy"] and \
        #    best_mf1 <= valid_outs["test/f1_score"]:
        #     best_acc = valid_outs["test/accuracy"]
        #     best_mf1 = valid_outs["test/f1_score"]
        #     update_epoch = epoch+1
        #     model.save_best_checkpoint(name="best_model")
        # if best_mf1 < valid_outs["test/f1_score"]:
        #     best_mf1 = valid_outs["test/f1_score"]
        #     update_epoch = epoch+1
        #     model.save_best_checkpoint(name="best_model")

        # Confusion matrix
        # if (epoch+1) % config["evaluate_span"] == 0 or (epoch+1) == config["n_epochs"]:
        #     logger.info(">> Confusion Matrix")
        #     logger.info(test_outs["test/cm"])
        #
        # # Save checkpoint
        # if (epoch+1) % config["checkpoint_span"] == 0 or (epoch+1) == config["n_epochs"]:
        #     model.save_checkpoint(name="model")
        #
        # # Early stopping
        # if update_epoch > 0 and ((epoch+1) - update_epoch) > config["no_improve_epochs"]:
        #     logger.info("*** Early-stopping ***")
        #     break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--fold_idx", type=int, required=True)
    parser.add_argument("--output_dir", type=str, default="./output/train")
    parser.add_argument("--restart", dest="restart", action="store_true")
    parser.add_argument("--no-restart", dest="restart", action="store_false")
    parser.add_argument("--log_file", type=str, default="./output/output.log")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.set_defaults(restart=False)
    args = parser.parse_args()

    train(
        config_file=args.config_file,
        fold_idx=args.fold_idx,
        output_dir=args.output_dir,
        log_file=args.log_file,
        restart=args.restart,
        random_seed=args.random_seed,
    )
