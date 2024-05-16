"""
Author: Anna M. Puszkarska
SPDX-License-Identifier: Apache-2.0
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

from peptide_models.aminoacids import AMINOACIDS
from peptide_models.utils_data import fasta2df, get_peptides, reformat_frame
from peptide_models.utils_models import get_predictions


def main(models_path: Path,
         output_path: Path,
         data_folder: Path):
    """
    Peptide potency prediction by the ensemble of
    multitask convolutional neural network models.
    Parameters:
    :param models_path: path to pre-trained models
    :param output_path: path to output the results
    :param data_folder: path to the folder with
    FASTA files
    :return:
    """
    start = time.time()
    # creates folder to store results
    if output_path.exists():
        raise IsADirectoryError(
            "Directory {} exists. Please remove.".format(str(output_path)))
    if not output_path.exists():
        output_path.mkdir(parents=True)
        print('Created path to store data.')

    for idx, fasta_file in enumerate(sorted(data_folder.iterdir())):
        fasta_data_path = str(fasta_file)
        data = fasta2df(data_path=fasta_data_path)
        print("Loading data ...")
        peptides = get_peptides(df=data)
        X = np.asarray([p.encoded_onehot for p in peptides])
        X_test = X.reshape(X.shape[0], X.shape[1], len(AMINOACIDS))
        print(f"Data set size:{len(X_test)}")

        print("Loading multi-task convolutional models ...")
        y_predicted = get_predictions(path_to_models=models_path, X_test=X_test)

        temp = np.asarray(y_predicted)
        temp1 = np.reshape(temp, (temp.shape[0], temp.shape[1], temp.shape[2]))
        d1 = reformat_frame(temp1, ['model', 'target', 'sample'])

        print("Computing mean ...")
        predictions = np.asarray(y_predicted).mean(0)
        predictionsR1 = predictions[0]
        predictionsR2 = predictions[1]

        print("Saving files with predictions ...")

        data = [(x, y, z, q, p, w) for x, y, z, q, p, w in
                zip(data.pep_ID.values, data.alias.values,
                    predictionsR1.flatten(),
                    predictionsR2.flatten(),
                    data.sequence.values,
                    data.length.values)]
        labels = ['pep_ID', 'alias',
                  'predicted_potency_GCGR',
                  'predicted_potency_GLP-1R',
                  'sequence',
                  'length']
        d2 = pd.DataFrame.from_records(data, columns=labels)

        name_file = fasta_data_path.split("/")[-1].split('.fasta')[0]

        with pd.ExcelWriter(str(Path(output_path,
                                     name_file).with_suffix('.xlsx'))) as writer:
            d2.to_excel(writer, sheet_name='ensemble_predictions',
                        index=True, float_format='%.4f')
            d1.to_excel(writer, sheet_name='predictions_models',
                        index=True, float_format='%.4f')
        end = time.time()
        print("Total time:{} min".format((end - start) / 60))


data_path = Path('data/FASTA_files/NCBI_data')
path_to_models = Path('models')

print('Running in {}'.format(str(Path.cwd())))

# specify your output folder
OUT_FOLDER = 'NCBI_data'
out_path = Path(Path.cwd(), 'results', 'predictions', OUT_FOLDER)
