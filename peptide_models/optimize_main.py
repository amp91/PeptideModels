"""
Author: Anna M. Puszkarska
SPDX-License-Identifier: Apache-2.0
"""

import time
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np

from peptide_models.aminoacids import AMINOACIDS
from peptide_models.peptide import Peptide
from peptide_models.utils_data import fasta2df, get_peptides, pep2fasta, peptides2df, save_frame
from peptide_models.utils_models import get_predictions


def select_sequences(peptides: List[Peptide],
                     low_th: Optional[float] = -9,
                     high_th: Optional[float] = -11,
                     dual_th: Optional[float] = -11.5) -> Tuple[List, List, List]:
    """
    Select sequences based on the potency criteria
    :param high_th: potency threshold (max)
    :param low_th: potency threshold (min)
    :param dual_th: potency threshold for dual
    :param peptides: input list of peptides
    :return: lists of peptides grouped by the potency
    """
    (gcgr_selective, glp1_r_selective,
     both_r_high_potency, non_active) = [], [], [], []
    for p in peptides:
        if (p.ec_50A >= low_th) & (p.ec_50B <= high_th):
            glp1_r_selective.append(p)
        elif (p.ec_50A <= high_th) & (p.ec_50B >= low_th):
            gcgr_selective.append(p)
        elif (p.ec_50A <= dual_th) & (p.ec_50B <= dual_th):
            both_r_high_potency.append(p)
        elif (p.ec_50A >= low_th) & (p.ec_50B >= low_th):
            non_active.append(p)

    print(f"Total no. of sequences :{len(peptides)}")
    print(f"Found {len(both_r_high_potency)} "
          f"peptides with high potency at both receptors.")
    print(f"Found {len(gcgr_selective)} "
          f"peptides with high potency against hGCGR.")
    print(f"Found {len(glp1_r_selective)} "
          f"peptides with high potency against hGLP-1R.")
    print(f"Found {len(non_active)} non-active peptides.")

    return gcgr_selective, glp1_r_selective, both_r_high_potency


def get_seeds(peptides_list: List[Peptide],
              number_of_seeds: int,
              number_of_sequences_to_save: int,
              output_path_preds: Path,
              output_path_fasta: Path) -> List[Peptide]:
    """
    :param peptides_list: list of peptides
    :param number_of_sequences_to_save: number of sequences
    to save (in FASTA format)
    :param number_of_seeds: number of seed peptides
    for optimization
    :param output_path_fasta: path to output FASTA file
    :param output_path_preds: path to output
    models predictions
    :return:
    """
    names = ['selective_towards_GCGR',
             'selective_towards_GLP1R',
             'high_potency_at_both']
    peptides_grouped = dict(zip(names,
                                select_sequences(peptides_list)))

    sg1, sg2, sg3 = [], [], []
    for key in peptides_grouped.keys():
        pep2fasta(peptides_list=peptides_grouped[key],
                  output_path=output_path_fasta,
                  dataset_name=key)
        df = peptides2df(peptides_grouped[key])
        if key == 'selective_towards_GCGR':
            df.sort_values(by='EC50_LOG_T1',
                           ascending=True,
                           inplace=True)
            # selects the best sequences to save
            num_seq_to_save = min(number_of_sequences_to_save, len(df))
            df = df.head(num_seq_to_save)
            num_seeds = min(number_of_seeds, len(df))
            # saves the best sequences
            save_frame(frame=df,
                       file_name=key,
                       out_path=output_path_preds)
            # generates seed sequences for the next round of optimization
            sg1 = get_peptides(df[:num_seeds])
        elif key == 'selective_towards_GLP1R':
            df.sort_values(by='EC50_LOG_T2',
                           ascending=True,
                           inplace=True)
            # select the best sequences to save
            num_seq_to_save = min(number_of_sequences_to_save, len(df))
            df = df.head(num_seq_to_save)
            num_seeds = min(number_of_seeds, len(df))
            # selected seed sequences for the next round of optimization
            sg2 = get_peptides(df[:num_seeds])
            save_frame(frame=df,
                       file_name=key,
                       out_path=output_path_preds)
        elif key == 'high_potency_at_both':
            df.sort_values(by='EC50_LOG_T1',
                           ascending=True,
                           inplace=True)
            # select the best sequences to save
            num_seq_to_save = min(number_of_sequences_to_save, len(df))
            df = df.head(num_seq_to_save)
            # selected seed sequences for the next round of optimization
            num_seeds = min(number_of_seeds, len(df))
            sg3 = get_peptides(df[:num_seeds])
            save_frame(frame=df,
                       file_name=key,
                       out_path=output_path_preds)
    return sg1 + sg2 + sg3


def optimize_seq(path_to_models: Path,
                 pep_training_set: List[Peptide],
                 samples_: List[Peptide],
                 out_path_fasta_: Path,
                 k: int) -> List[Peptide]:
    """
    :param k: generation index
    :type path_to_models: path to pre-trained models
    :param out_path_fasta_: path to output FASTA file
    :param pep_training_set: training set peptides
    :param samples_: samples
    :return:
    """
    # get point mutants
    point_mutants = []
    for pep_ in samples_:
        point_mutants_current = pep_.get_single_point_mutants()
        point_mutants.append(point_mutants_current)
    point_mutants = [item for sublist in point_mutants for item in sublist]

    seq_to_reject = [pm.sequence for pm in pep_training_set]
    seqs_seen = []
    for seq in point_mutants:
        # Remove redundant sequences
        if seq not in seqs_seen:
            # Check if not in the training set
            if seq not in seq_to_reject:
                seqs_seen.append(seq)

    print(f"Training set size:{len(pep_training_set)},"
          f"\nInitial number of samples:{len(point_mutants)},"
          f"\nNumber of repetitions:{(len(point_mutants) - len(seqs_seen))},"
          f"\nNumber of samples in the {k} generation:{len(seqs_seen)}")
    # generate peptides from samples
    generic_peptides = []
    for idx, sequence in enumerate(seqs_seen):
        peptide = Peptide(alias='sample',
                          ec_50A=None,
                          ec_50B=None,
                          name=idx,
                          sequence=sequence,
                          c_term=True)
        generic_peptides.append(peptide)

    # save FASTA files with 1000 example samples
    num_to_save = min(1000, len(generic_peptides))
    pep2fasta(peptides_list=generic_peptides[:num_to_save],
              output_path=out_path_fasta_,
              dataset_name='samples')
    # predict potencies for samples
    X = np.asarray([p.encoded_onehot for p in generic_peptides])
    X_test = X.reshape(X.shape[0], X.shape[1], len(AMINOACIDS))
    print(f"Data set size:{len(X_test)}")
    print("Loading models...")
    y_predicted = get_predictions(path_to_models=path_to_models,
                                  X_test=X_test)
    # average ensemble prediction
    print("Predicting for samples ...")
    predictions = np.asarray(y_predicted).mean(0)
    predictionsR1 = predictions[0]
    predictionsR2 = predictions[1]
    for i, sample_peptide in enumerate(generic_peptides):
        sample_peptide.ec_50A = predictionsR1[i][0]
        sample_peptide.ec_50B = predictionsR2[i][0]
    return generic_peptides


def main(training_data_path: Path,
         path_to_trained_models: Path,
         out_path_fasta_files: Path,
         out_path_ensemble_predictions: Path,
         num_seeds: int,
         num_generations: int,
         num_seqs_to_save: int) -> None:
    """
    Peptide Sequence Optimization by the Model-Guided
    Directed Evolution Algorithm.
    Parameters:
    :param num_seqs_to_save: number of the best sequences to save
    :param num_seeds: number of seed sequences in each run
    :param num_generations: number of optimization runs
    :param training_data_path: path to the initial set of sequences
    :param path_to_trained_models: path to the pre-trained models
    :param out_path_fasta_files: path to store FASTA files with samples
    :param out_path_ensemble_predictions: path to store predictions
    :return:
    """
    if not out_path_fasta_files.exists():
        out_path_fasta_files.mkdir(parents=True)
        print('Created path to store data.')

    if not out_path_ensemble_predictions.exists():
        out_path_ensemble_predictions.mkdir(parents=True)
        print('Created path to store data.')

    start = time.time()
    # reads training data
    data = fasta2df(data_path=str(training_data_path))
    print("Loading data ...")
    training_peptides = get_peptides(df=data)

    path_predictions_gen0 = Path(out_path_ensemble_predictions, 'gen_0')
    path_FASTA_gen0 = Path(out_path_fasta_files, 'gen_0')

    if not path_predictions_gen0.exists():
        path_predictions_gen0.mkdir(parents=True)
        print('Created path to store data.')

    if not path_FASTA_gen0.exists():
        path_FASTA_gen0.mkdir(parents=True)
        print('Created path to store data.')

    samples = training_peptides
    sample_peptides = optimize_seq(path_to_models=path_to_trained_models,
                                   pep_training_set=training_peptides,
                                   samples_=samples,
                                   k=0,
                                   out_path_fasta_=path_FASTA_gen0)

    seeds = get_seeds(peptides_list=sample_peptides,
                      number_of_seeds=num_seeds,
                      number_of_sequences_to_save=num_seqs_to_save,
                      output_path_preds=path_predictions_gen0,
                      output_path_fasta=path_FASTA_gen0)

    for G in range(1, num_generations):
        print(f"Generates set of sequences with {G} "
              f"step mutations from each training set sequence.")
        output_preds_gen = Path(out_path_ensemble_predictions, f'gen_{G}')
        output_fasta_gen = Path(out_path_fasta_files, f'gen_{G}')
        if not output_preds_gen.exists():
            output_preds_gen.mkdir(parents=True)
            print('Created path to store peptide optimization data.')
        if not output_fasta_gen.exists():
            output_fasta_gen.mkdir(parents=True)
            print('Created path to store peptide optimization data.')

        sample_peptides = optimize_seq(path_to_models=path_to_trained_models,
                                       pep_training_set=training_peptides,
                                       samples_=seeds,
                                       out_path_fasta_=output_fasta_gen,
                                       k=G)
        seeds_new = get_seeds(peptides_list=sample_peptides,
                              output_path_preds=output_preds_gen,
                              output_path_fasta=output_fasta_gen,
                              number_of_sequences_to_save=num_seqs_to_save,
                              number_of_seeds=num_seeds)
        seeds = seeds_new

    end = time.time()
    print("Total time:{} min".format((end - start) / 60))
    return


path2training_data = Path('data/FASTA_files/training_data_msa.fasta')
path2models = Path('models')
out_path_FASTA = Path('results', 'ligand_design', 'samples', 'FASTA_files')
out_path_predictions = Path('results', 'ligand_design', 'samples', 'predictions')

NUM_SEEDS = 10
NUM_GENERATIONS = 3
NUM_SEQ_TO_SAVE = 50

if __name__ == '__main__':
    main(training_data_path=path2training_data,
         path_to_trained_models=path2models,
         out_path_ensemble_predictions=out_path_predictions,
         out_path_fasta_files=out_path_FASTA,
         num_seeds=NUM_SEEDS,
         num_generations=NUM_GENERATIONS,
         num_seqs_to_save=NUM_SEQ_TO_SAVE)