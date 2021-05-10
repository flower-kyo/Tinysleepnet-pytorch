import argparse
import glob
import importlib
import os
import numpy as np
import sklearn.metrics as skmetrics
import torch

from models.model_tinysleepnet import Model

from dataTools.data import load_data, get_subject_files
from dataTools.minibatching import (iterate_batch_multiple_seq_minibatches)
from script.utils import (print_n_samples_each_class,
                          load_seq_ids)
from script.logger import get_logger


def compute_performance(cm):
    """Computer performance metrics from confusion matrix.

    It computers performance metrics from confusion matrix.
    It returns:
        - Total number of samples
        - Number of samples in each class
        - Accuracy
        - Macro-F1 score
        - Per-class precision
        - Per-class recall
        - Per-class f1-score
    """

    tp = np.diagonal(cm).astype(np.float)
    tpfp = np.sum(cm, axis=0).astype(np.float) # sum of each col
    tpfn = np.sum(cm, axis=1).astype(np.float) # sum of each row
    acc = np.sum(tp) / np.sum(cm)
    precision = tp / tpfp
    recall = tp / tpfn
    f1 = (2 * precision * recall) / (precision + recall)
    mf1 = np.mean(f1)

    total = np.sum(cm)
    n_each_class = tpfn

    return total, n_each_class, acc, mf1, precision, recall, f1


def predict(
    config_file,
    model_dir,
    output_dir,
    log_file,
    use_best=True,
):
    spec = importlib.util.spec_from_file_location("*", config_file)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    config = config.predict

    # Create output directory for the specified fold_idx
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create logger
    logger = get_logger(log_file, level="info")

    subject_files = glob.glob(os.path.join(args.data_dir, "*.npz"))

    # Load subject IDs
    fname = "./config/{}.txt".format(config["dataset"])
    seq_sids = load_seq_ids(fname)
    logger.info("Load generated SIDs from {}".format(fname))
    logger.info("SIDs ({}): {}".format(len(seq_sids), seq_sids))

    # Split training and test sets
    fold_pids = np.array_split(seq_sids, config["n_folds"])

    # Add dummy class weights
    config["class_weights"] = np.ones(config["n_classes"], dtype=np.float32)

    trues = []
    preds = []
    for fold_idx in range(config["n_folds"]):

        logger.info("------ Fold {}/{} ------".format(fold_idx+1, config["n_folds"]))

        test_sids = fold_pids[fold_idx]
        rep_sids = np.setdiff1d(seq_sids, test_sids)

        logger.info("Test SIDs: ({}) {}".format(len(test_sids), test_sids))
        device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
        # Get corresponding files
        s_trues = []
        s_preds = []
        for sid in test_sids:  # test one by one
            logger.info("loading trained model")
            model = Model(
                config=config,
                output_dir=os.path.join(model_dir, str(fold_idx)),
                use_rnn=True,
                testing=True,
                use_best=use_best,
                device=device
            )

            # get representation of all train and valid subjects
            reps = []
            labels = []
            subjects = []
            for r_sid in rep_sids:
                logger.info("loading represent Subject ID: {}".format(r_sid))
                rep_file = get_subject_files(
                    dataset=config["dataset"],
                    files=subject_files,
                    sid=r_sid,
                )
                rep_x, rep_y, _ = load_data(rep_file)
                for night_idx, night_data in enumerate(zip(rep_x, rep_y)):
                    night_x, night_y = night_data
                    test_minibatch_fn = iterate_batch_multiple_seq_minibatches(
                        [night_x],
                        [night_y],
                        batch_size=config["batch_size"],
                        seq_length=config["seq_length"],
                        shuffle_idx=None,
                        augment_seq=False,
                    )
                    rep_outs = model.get_rep(test_minibatch_fn)
                    reps.append(rep_outs)
                    labels.append(torch.from_numpy(night_y))
                    subjects.append(torch.ones(size=(night_y.shape[0],)) * r_sid)
            reps = torch.cat(reps)
            labels = torch.cat(labels)
            subjects = torch.cat(subjects)

            # get representation of test subject
            logger.info("get rep of test Subject ID: {}".format(sid))
            test_files = get_subject_files(
                dataset=config["dataset"],
                files=subject_files,
                sid=sid,
            )
            test_x, test_y, _ = load_data(test_files)  # only one test subject once
            test_reps = []
            test_labels = []
            test_subjects = []
            for night_idx, night_data in enumerate(zip(test_x, test_y)):
                night_x, night_y = night_data
                test_minibatch_fn = iterate_batch_multiple_seq_minibatches(
                    [night_x],
                    [night_y],
                    batch_size=config["batch_size"],
                    seq_length=config["seq_length"],
                    shuffle_idx=None,
                    augment_seq=False,
                )
                rep_outs = model.get_rep(test_minibatch_fn)
                test_reps.append(rep_outs)
                test_labels.append(torch.from_numpy(night_y))
                test_subjects.append(torch.ones(size=(night_y.shape[0],)) * r_sid)
            test_reps = torch.cat(test_reps)
            test_labels = torch.cat(test_labels)
            test_subjects = torch.cat(test_subjects)

            # from train and valid subjects, find the subjects most similar to test subjdect
            similar_x, similar_y = get_similar()  # todo
            # finetune model with similar_x, similar_y
            # test













            logger.info("testing on Subject ID: {}".format(sid))
            test_files = get_subject_files(
                dataset=config["dataset"],
                files=subject_files,
                sid=sid,
            )
            for vf in test_files: logger.info("Load files {} ...".format(vf))

            test_x, test_y, _ = load_data(test_files)

            # Print test set
            logger.info("Test set (n_night_sleeps={})".format(len(test_y)))
            for _x in test_x: logger.info(_x.shape)
            print_n_samples_each_class(np.hstack(test_y))

            for night_idx, night_data in enumerate(zip(test_x, test_y)):
                # Create minibatches for testing
                night_x, night_y = night_data
                test_minibatch_fn = iterate_batch_multiple_seq_minibatches(
                    [night_x],
                    [night_y],
                    batch_size=config["batch_size"],
                    seq_length=config["seq_length"],
                    shuffle_idx=None,
                    augment_seq=False,
                )
                # Evaluate
                test_outs = model.evaluate_with_dataloader(test_minibatch_fn)  # 预测入口在这里
                s_trues.extend(test_outs["test/trues"])
                s_preds.extend(test_outs["test/preds"])
                trues.extend(test_outs["test/trues"])
                preds.extend(test_outs["test/preds"])

                # Save labels and predictions (each night of each subject)
                save_dict = {
                    "y_true": test_outs["test/trues"],
                    "y_pred": test_outs["test/preds"],
                }
                fname = os.path.basename(test_files[night_idx]).split(".")[0]
                save_path = os.path.join(
                    output_dir,
                    "pred_{}.npz".format(fname)
                )
                np.savez(save_path, **save_dict)
                logger.info("Saved outputs to {}".format(save_path))

        s_acc = skmetrics.accuracy_score(y_true=s_trues, y_pred=s_preds)
        s_f1_score = skmetrics.f1_score(y_true=s_trues, y_pred=s_preds, average="macro")
        s_cm = skmetrics.confusion_matrix(y_true=s_trues, y_pred=s_preds, labels=[0,1,2,3,4])

        logger.info("n={}, acc={:.1f}, mf1={:.1f}".format(
            len(s_preds),
            s_acc*100.0,
            s_f1_score*100.0,
        ))

        logger.info(">> Confusion Matrix")
        logger.info(s_cm)


        logger.info("------------------------")
        logger.info("")

    acc = skmetrics.accuracy_score(y_true=trues, y_pred=preds)
    f1_score = skmetrics.f1_score(y_true=trues, y_pred=preds, average="macro")
    cm = skmetrics.confusion_matrix(y_true=trues, y_pred=preds, labels=[0,1,2,3,4])

    logger.info("")
    logger.info("=== Overall ===")
    print_n_samples_each_class(trues)
    logger.info("n={}, acc={:.1f}, mf1={:.1f}".format(
        len(preds),
        acc*100.0,
        f1_score*100.0,
    ))

    logger.info(">> Confusion Matrix")
    logger.info(cm)

    metrics = compute_performance(cm=cm)
    logger.info("Total: {}".format(metrics[0]))
    logger.info("Number of samples from each class: {}".format(metrics[1]))
    logger.info("Accuracy: {:.1f}".format(metrics[2]*100.0))
    logger.info("Macro F1-Score: {:.1f}".format(metrics[3]*100.0))
    logger.info("Per-class Precision: " + " ".join(["{:.1f}".format(m*100.0) for m in metrics[4]]))
    logger.info("Per-class Recall: " + " ".join(["{:.1f}".format(m*100.0) for m in metrics[5]]))
    logger.info("Per-class F1-Score: " + " ".join(["{:.1f}".format(m*100.0) for m in metrics[6]]))

    # Save labels and predictions (all)
    save_dict = {
        "y_true": trues,
        "y_pred": preds,
        "seq_sids": seq_sids,
        "config": config,
    }
    save_path = os.path.join(
        output_dir,
        "{}.npz".format(config["dataset"])
    )
    np.savez(save_path, **save_dict)
    logger.info("Saved summary to {}".format(save_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--model_dir", type=str, default="./out_sleepedf/finetune")
    parser.add_argument("--output_dir", type=str, default="./output/predict")
    parser.add_argument("--log_file", type=str, default="./output/output.log")
    parser.add_argument("--use-best", dest="use_best", action="store_true")
    parser.add_argument("--no-use-best", dest="use_best", action="store_false")
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--data_dir", type=str, default='../tinysleepnet/data/sleepedf/sleep-cassette/eeg_fpz_cz')
    parser.set_defaults(use_best=False)
    args = parser.parse_args()

    predict(
        config_file=args.config_file,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        log_file=args.log_file,
        use_best=args.use_best,
    )
