"""
Author: Anna M. Puszkarska
SPDX-License-Identifier: Apache-2.0
"""
import multiprocessing as mp
import time
from pathlib import Path
from typing import Tuple, List

import keras.backend as K
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from peptide_models.aminoacids import AMINOACIDS
from peptide_models.peptide import Peptide
from peptide_models.utils_models import build_model, fit_model
from peptide_models.utils_plotting import get_voting_reg_plot, \
    get_training_curves, get_plot


def train_ensemble(num_models: int,
                   X_train: np.ndarray,
                   y_train: List,
                   x_test: np.ndarray,
                   y_test: List,
                   x_validation: np.ndarray,
                   y_validation: List,
                   in_shape: Tuple[int, int],
                   path_figures: Path,
                   path_to_models: Path,
                   ) -> np.ndarray:
    """
    Ensemble model training.
    :param num_models: number of models in the ensemble
    :param X_train: one-hot encoded sequences (training)
    :param y_train: target values (training)
    :param x_test: one-hot encoded sequences (test set)
    :param y_test: target values (test)
    :param x_validation: one-hot encoded sequences
    (validation set)
    :param y_validation: target values (validation set)
    :param in_shape: Conv1D layer input shape
    [sequence length x encoding dimension]
    :param path_figures: path to safe figures
    :param path_to_models:path to safe trained models
    :return: ensemble model prediction
    """
    predictions_ensemble = []
    for index in range(num_models):
        model = build_model(in_shape=in_shape)
        history = fit_model(model=model,
                            x_train=X_train,
                            y_train=y_train,
                            max_epochs=1500,
                            val_data=(x_validation, y_validation))

        get_training_curves(out_path=path_figures,
                            model_history=history,
                            name=f'training_curve{index}', s=True)

        print(f"Metric names:{model.metrics_names}")

        print("\n Evaluate on training data")
        evaluation_train = model.evaluate(X_train, y_train,
                                          batch_size=25, verbose=False)
        print(f"training loss, training accuracy:{evaluation_train}")

        print("\n Evaluate on test data")
        evaluation_test = model.evaluate(x_test, y_test, verbose=False)
        print(f"test loss, test accuracy:{evaluation_test}")

        prediction = model.predict(x_test)
        predictions_ensemble.append(prediction)

        model.save(str(Path(path_to_models,
                            f'multi-task_model{index}').with_suffix(".h5")))

    pred_1 = np.asarray(predictions_ensemble)[:, 0]
    pred_2 = np.asarray(predictions_ensemble)[:, 1]

    av_vote = np.asarray(predictions_ensemble).mean(axis=0)

    get_voting_reg_plot(pred_1, av_vote[0], np.asarray(y_test)[0],
                        name='receptor1', s=True, out_path=path_figures)
    get_voting_reg_plot(pred_2, av_vote[1], np.asarray(y_test)[1],
                        name='receptor2', s=True, out_path=path_figures)

    return av_vote


def train_multitask_models(iteration: int,
                           train_indexes: np.ndarray,
                           test_indexes: np.ndarray,
                           path_to_models: Path,
                           path_figures: Path,
                           X: np.ndarray,
                           y: np.ndarray):
    NUM_MODELS = 12  # number of models in the ensemble
    # training data
    X_current = X[train_indexes]
    X_current = X_current.reshape(X_current.shape[0],
                                  X_current.shape[1],
                                  len(AMINOACIDS))
    y1_current = y[0][train_indexes]
    y2_current = y[1][train_indexes]
    y_current = [y1_current, y2_current]
    # validation/test split
    test_indexes = test_indexes[:int(len(test_indexes) / 2)]
    val_indexes = test_indexes[int(len(test_indexes) / 2):]

    X_validation = X[val_indexes]
    X_validation = X_validation.reshape(X_validation.shape[0],
                                        X_validation.shape[1],
                                        len(AMINOACIDS))
    y1_validation = y[0][val_indexes]
    y2_validation = y[1][val_indexes]
    y_validation = [y1_validation, y2_validation]

    X_test = X[test_indexes]
    X_test = X_test.reshape(X_test.shape[0],
                            X_test.shape[1],
                            len(AMINOACIDS))
    y1_test = y[0][test_indexes]
    y2_test = y[1][test_indexes]
    y_test = [y1_test, y2_test]

    in_shape = (X_current.shape[1], X_current.shape[2])

    y_hat = train_ensemble(num_models=NUM_MODELS,
                           X_train=X_current,
                           y_train=y_current,
                           x_validation=X_validation,
                           x_test=X_test,
                           y_test=y_test,
                           y_validation=y_validation,
                           in_shape=in_shape,
                           path_figures=path_figures,
                           path_to_models=path_to_models)

    get_plot(y_test=y_test,
             y_pred=y_hat,
             path_to_figs=path_figures,
             name=str(iteration))

    K.clear_session()

    return y_hat, y_test


def main(training_data_path: Path,
         seed: int,
         out_path: Path):
    NUM_FOLDS = 6  # number of cross-validation folds
    start = time.time()

    if out_path.exists():
        raise IsADirectoryError(
            f"Directory {str(out_path)} exists. Please remove.")
    if not out_path.exists():
        out_path.mkdir(parents=True)
        print('Created path to store data.')

    # print('Running in {}'.format(str(Path.cwd())))

    # reads training data
    dataset = pd.read_excel(str(training_data_path),
                            index_col=0,
                            header=0,
                            skiprows=0,
                            sheet_name='dataset')
    msa = pd.read_excel(str(training_data_path),
                        index_col=0,
                        header=0,
                        skiprows=0,
                        sheet_name='alignment')

    training_peptides = []
    for idx in range(len(dataset)):
        pep_record = dataset.iloc[idx]
        peptide = Peptide(alias=pep_record.alias,
                          ec_50A=pep_record.EC50_LOG_T1,
                          ec_50B=pep_record.EC50_LOG_T2,
                          name=idx,
                          sequence=msa.iloc[idx].sequence,
                          c_term=True)
        training_peptides.append(peptide)

    X = np.asarray([p.encoded_onehot for p in training_peptides])
    n_train_examples, seq_length, n_categories = X.shape
    print(f"Number of examples in the training set:{n_train_examples}")
    print(f"Number of sequence positions (features):{seq_length}")
    print(f"Dimensionality of categorical encoding:{n_categories}")
    y1 = [p.ec_50A for p in training_peptides]
    y2 = [p.ec_50B for p in training_peptides]
    y = np.asarray([y1, y2])

    kf = KFold(n_splits=NUM_FOLDS,
               shuffle=True,
               random_state=seed)

    tasks = []
    for idx, (train_index, test_index) in enumerate([j for j in kf.split(X)]):

        # creates folder to store trained models
        path_models = Path(out_path, f'models{idx}')
        # creates folder to store figures
        path_figs = Path(out_path, f'figs_models{idx}')

        if not path_models.exists():
            path_models.mkdir(parents=True)
            print('Created path to store models.')

        if not path_figs.exists():
            path_figs.mkdir(parents=True)
            print('Created path to store figures.')

        print(f"Iteration {idx + 1}/{NUM_FOLDS}\n"
              f"Training set size:{len(train_index)},"
              f"Test set size:{len(test_index)}")

        tasks.append((idx, train_index, test_index, path_models, path_figs, X, y))

    with mp.Pool(16) as pool:

        results = pool.starmap(train_multitask_models, tasks)

    predictions1 = []
    predictions2 = []
    targets1 = []
    targets2 = []

    for j in range(NUM_FOLDS):
        y_hat_current, y_val = results[j]
        predictions1.append(y_hat_current[0])
        predictions2.append(y_hat_current[1])
        targets1.append(y_val[0])
        targets2.append(y_val[1])

    predictions1 = [item for sublist in predictions1 for item in sublist]
    predictions2 = [item for sublist in predictions2 for item in sublist]
    targets2 = [item for sublist in targets2 for item in sublist]
    targets1 = [item for sublist in targets1 for item in sublist]

    data = {'GCGR_true_y': np.asarray(targets1).flatten(),
            'GCGR_predicted_y': np.asarray(predictions1).flatten(),
            'GLP-1R_true_y': np.asarray(targets2).flatten(),
            'GLP-1R_predicted_y': np.asarray(predictions2).flatten()}
    df = pd.DataFrame(data)

    name_xlsx = Path(out_path, 'predictions_multi-task').with_suffix('.xlsx')

    with pd.ExcelWriter(name_xlsx) as writer:
        df.to_excel(writer, sheet_name='predicted',
                    index=True, float_format='%.4f')

    end = time.time()
    print("Total time:{} min".format((end - start) / 60))


SEED = 21
data_path = Path('data/training_data.xlsx')
output_path = Path("results", 'training')

if __name__ == '__main__':
    main(training_data_path=data_path,
         out_path=output_path,
         seed=SEED)
