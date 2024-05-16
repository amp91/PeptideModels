"""
Author: Anna M. Puszkarska
SPDX-License-Identifier: Apache-2.0
"""

import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from pathlib import Path
from typing import List

from peptide_models.peptide import Peptide


def df2fasta(df: pd.DataFrame,
             dataset_name: str) -> None:
    """
    Reads pandas data frame
    with sequence records
    and saves FASTA file
    :param dataset_name: name of the dataset
    :param df: frame with data
    """
    df.index = range(1, len(df) + 1)
    # make FASTA file
    records = []
    for row in df.T:
        try:
            record = SeqRecord(Seq(df.loc[row].sequence),
                               id=str(df.loc[row].alias),
                               description=f'[{dataset_name}]')
            records.append(record)
        except AttributeError as error:
            print('Wrong file!', error)
            raise

    SeqIO.write(records, f'{dataset_name}.fasta', "fasta")


def get_peptides(df: pd.DataFrame) -> List[Peptide]:
    """
    Reads pandas data frame and coverts it to
    the list of peptides
    :param df: input data frame with
    sequence records
    :return: peptide list
    """
    peptides = []
    for alias in range(len(df)):
        pep_record = df.iloc[alias]
        try:
            peptide = Peptide(alias=pep_record.pep_ID,
                              name=alias,
                              sequence=pep_record.sequence,
                              c_term=True)
            peptides.append(peptide)
        except AttributeError as error:
            print('Wrong data frame!', error)
            raise error
        pass

    return peptides


def pep2fasta(peptides_list: List[Peptide],
              output_path: Path,
              dataset_name: str) -> None:
    """
    Writes FASTA file from a list of peptides
    :param peptides_list: list of peptides
    :param output_path: path to store FASTA
    :param dataset_name: name of the data set
    :return:
    """
    records = []
    for pep_record in peptides_list:
        seq = "".join([a.letter for a in pep_record.seq_amino])
        record = SeqRecord(Seq(seq),
                           id=str(pep_record.alias),
                           name=str(pep_record.name),
                           description='peptide: ' + str(pep_record.name))
        records.append(record)

    SeqIO.write(records, str(Path(output_path,
                                  dataset_name).with_suffix('.fasta')), "fasta")


def fasta2df(data_path: str) -> pd.DataFrame:
    """
    Loads data from FASTA file and returns
    data frame
    :param data_path: path to fasta file
    :return: data frame -> data
    """
    with open(data_path) as fasta_file:
        identifiers = []
        lengths = []
        sequences = []
        names = []
        for seq_record in SeqIO.parse(fasta_file, 'fasta'):
            name = seq_record.description.split('[')[-1].split(']')[0]
            identifiers.append(seq_record.id)
            lengths.append(len(seq_record.seq))
            sequences.append(''.join(seq_record.seq))
            names.append(name)

    data = [(x, y, z, q) for x, y, z, q in zip(identifiers,
                                               names,
                                               sequences,
                                               lengths)]
    labels = ['pep_ID', 'alias', 'sequence', 'length']
    data = pd.DataFrame.from_records(data, columns=labels)

    return data


def reformat_frame(A, columns) -> pd.DataFrame:
    index = pd.MultiIndex.from_product([range(s) for s in A.shape], names=columns)
    df = pd.DataFrame({'prediction': A.flatten()}, index=index).reset_index()
    return df


def peptides2df(peptides: List[Peptide]) -> pd.DataFrame:
    """
    Converts list of Peptides into dataframe
    :param peptides: input list of Peptide
    instances
    :return: dataframe with records
    """
    try:
        list_pep = [[pep.name for pep in peptides],
                    [pep.alias for pep in peptides],
                    [''.join(list(pep.sequence)) for pep in peptides],
                    [pep.ec_50A for pep in peptides],
                    [pep.ec_50B for pep in peptides],
                    [pep.length for pep in peptides],
                    [pep.pi for pep in peptides],
                    [pep.molecular_weight for pep in peptides],
                    [pep.aromaticity for pep in peptides],
                    [pep.instability_index for pep in peptides],
                    [pep.gravy for pep in peptides],
                    [pep.m_ext_coefficient[0] for pep in peptides]]

        data = pd.DataFrame(data=list_pep)
        data = data.T
        data.rename(columns={x: y for x, y in zip(data.columns,
                                                  ['pep_ID', 'alias',
                                                   'sequence',
                                                   'EC50_LOG_T1',
                                                   'EC50_LOG_T2',
                                                   'length', 'pI',
                                                   'MW', 'A', 'II',
                                                   'G', 'M'])}, inplace=True)
        data.index = range(1, len(data) + 1)
    except AttributeError as error:
        print('Peptide lacks properties. Please check your data.', error)
        raise

    return data


def save_frame(frame: pd.DataFrame,
               out_path: Path,
               file_name: str) -> None:
    """
    Save dataframe
    :param frame: frame to save
    :param file_name: name of the file
    :type out_path: path to store the data
    :return:
    """
    name_xlsx = Path(out_path, file_name).with_suffix('.xlsx')
    frame.index = range(1, len(frame) + 1)

    with pd.ExcelWriter(name_xlsx) as writer:
        frame.to_excel(writer, sheet_name='predicted',
                       index=True, float_format='%.4f')
