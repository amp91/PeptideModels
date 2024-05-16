"""
Author: Anna M. Puszkarska

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from functools import total_ordering
from typing import List, Optional

import numpy as np
import tensorflow as tf
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from sklearn import preprocessing

from peptide_models.aminoacids import AMINOACIDS, LETTER2AA, AminoAcid


@total_ordering
class Peptide:
    properties = True

    def __init__(self,
                 sequence: str,
                 alias: str,
                 name: int,
                 c_term: bool,
                 ec_50A: Optional[float] = None,
                 ec_50B: Optional[float] = None,
                 phylo: Optional[str] = None,
                 ):
        """
        Initialize a Peptide object.
        :param sequence: string with peptide sequence
        :param alias: peptide identifier
        :param name: peptide index
        :param c_term: type of C-term; True - amide, False - acid
        :param ec_50A: log10EC50 at R1 (GCGR)
        :param ec_50B: log10EC50 at R2 (GLP1-R)
        :param phylo: Latin name of a species
        """
        self.sequence = sequence
        self.length = len(self.sequence)
        self.alias = alias
        self.name = name
        self.ec_50A = ec_50A
        self.ec_50B = ec_50B
        self.c_term = c_term
        self.seq_amino = self.convert_sequence()
        self.phylo = phylo

        if self.seq_amino:
            self.encoded_onehot = self.encode_one_hot()

        if self.properties:
            self.molecular_weight = self.get_seq().molecular_weight()
            self.amino_count = self.get_seq().count_amino_acids()
            self.aromaticity = self.get_seq().aromaticity()
            self.instability_index = self.get_seq().instability_index()
            self.m_ext_coefficient = self.get_seq().molar_extinction_coefficient()
            self.gravy = self.get_seq().gravy()
            self.pi = self.get_seq().isoelectric_point()

    def get_seq(self):
        seq = "".join([x if x not in ['-', 'B', 'X', 'Z']
                       else "" for x in self.sequence])
        seq = ProteinAnalysis(prot_sequence=seq)
        return seq

    def get_single_point_mutants(self) -> List[str]:
        """
        Generate a set of single point mutants
        :return: set of amino acid sequences as strings
        """
        letters = sorted([a.letter for a in AMINOACIDS[:-1]])
        root = [str(x) for x in self.sequence]
        samples = []
        for idx in range(len(root)):
            for aa in letters:
                sample = root.copy()
                aa_at_pos = sample[idx]
                if aa_at_pos != aa:
                    sample[idx] = aa
                    samples.append("".join(sample))
        return samples

    def __str__(self):
        return "Peptide object with sequence: {}, " \
               "alias: {}.".format(self.sequence, self.alias)

    def __eq__(self, other):
        return self.name == other.name

    def __ne__(self, other):
        return not self == other

    def __lt__(self, other):
        return self.name < other.name

    def convert_sequence(self) -> List[AminoAcid]:
        """
        Convert sequence (str) into list of amino acids
        :return: sequence
        """
        if np.sum([x.islower() for x in self.sequence]) >= 1:
            seq_aa = []
            print("Lower-case letters in the sequence!Cannot "
                  "convert.", self.sequence)
        else:
            seq_aa = []
            for x in self.sequence:
                if x in [a.letter for a in AMINOACIDS]:
                    val = LETTER2AA[x]
                    seq_aa.append(val)
        return seq_aa

    def encode_one_hot(self):
        """
        Encodes sequence as a binary array
        :return: array, dim: [len_sequence x 21]
        """

        def encode_aa():
            letters_amino = sorted([a.letter for a in AMINOACIDS])
            x = []
            for idx in range(len(self.seq_amino)):
                x.append(letters_amino)
            x = np.array(x).T
            encoding = preprocessing.OneHotEncoder(categories='auto',
                                                   handle_unknown='ignore')
            encoding.fit(x)
            return encoding

        seq_letters = [a.letter for a in self.seq_amino]
        enc = encode_aa()
        seq_one_hot = enc.transform(np.asarray(seq_letters)[None]).toarray()

        return seq_one_hot.reshape(len(self.sequence), len(AMINOACIDS))

    def predict_potency(self, infers: List[tf.keras.Model], num_targets=2):
        """
        Calls pre-trained ensemble of multitask neural network
        models and returns the average ensemble prediction
        :param num_targets: number of targets
        :param infers: list of pre-trained models
        :return:
        """
        q = len(infers)

        predictions_ensemble = np.zeros((q, num_targets))
        for i, model in enumerate(infers):
            d = self.encoded_onehot.reshape(
                (1, self.encoded_onehot.shape[0], self.encoded_onehot.shape[1]))
            predictions = model.predict(d, verbose=False)
            predictions = np.asarray(predictions).reshape(1, num_targets)
            predictions_ensemble[i, :] = predictions
        preds = predictions_ensemble.mean(axis=0)
        if self.ec_50A is None:
            self.ec_50A = preds[0]
        if self.ec_50B is None:
            self.ec_50B = preds[1]