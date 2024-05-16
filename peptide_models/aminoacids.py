class AminoAcid:
    def __init__(self,
                 letter: str,
                 index: int,
                 electrostatic: bool,
                 hydrophobic_e: float,
                 pka: float,
                 charge: float,
                 hydrophobic_kd: float,
                 iso_point: float,
                 bg_freq: float,
                 three_letter_code: str):
        """
        Initialize the AminoAcid.
        :param letter: single letter code
        :param index: index
        :param electrostatic: electrostatic flag
        :param hydrophobic_e: hydrophobicity
        scale: Eisenberg D., Schwarz E., Komarony M., Wall R.
        J. Mol. Biol. 179:125-142(1984).
        :param pka: pKa (side chain)
        :param charge: binary flag of the side chain charge
        :param hydrophobic_kd: hydropathicity
        scale: Kyte J., Doolittle R.F.
        J. Mol. Biol. 157:105-132(1982).
        :param iso_point: isoelectric point
        :param bg_freq: background frequency
        of amino acid occurrences in proteins:
        Amino acid composition form UniProt (2022)
        https://www.uniprot.org/uniprotkb/statistics
        :param three_letter_code: three-letter code of the
        amino acid
        """
        self.letter = letter
        self.index = index
        self.electrostatic = electrostatic
        self.hydrophobic_e = hydrophobic_e
        self.pka = pka
        self.charge = charge
        self.hydrophobic_kd = hydrophobic_kd
        self.iso_point = iso_point
        self.bg_freq = bg_freq
        self.three_letter_code = three_letter_code

    def __repr__(self):
        return self.letter


C = AminoAcid('C', 0, False, 0.29, 8.5, 0, 2.5, 5.07, 0.0120, 'CYS')
M = AminoAcid('M', 1, False, 0.64, 0, 0, 1.9, 5.74, 0.0238, 'MET')
F = AminoAcid('F', 2, False, 1.19, 0, 0, 2.8, 5.48, 0.0392, 'PHE')
I = AminoAcid('I', 3, False, 1.38, 0, 0, 4.5, 6.02, 0.0567, 'ILE')
L = AminoAcid('L', 4, False, 1.06, 0, 0, 3.8, 5.98, 0.0990, 'LEU')
V = AminoAcid('V', 5, False, 1.08, 0, 0, 4.2, 5.96, 0.0690, 'VAL')
W = AminoAcid('W', 6, False, 0.81, 0, 0, -0.9, 5.89, 0.0130, 'TRP')
Y = AminoAcid('Y', 7, False, 0.26, 10.1, 0, -1.3, 5.66, 0.0291, 'TYR')
A = AminoAcid('A', 8, False, 0.62, 0, 0, 1.8, 6.00, 0.0917, 'ALA')
G = AminoAcid('G', 9, False, 0.48, 0, 0, -0.4, 5.97, 0.0733, 'GLY')
T = AminoAcid('T', 10, False, -0.05, 0, 0, -0.7, 5.60, 0.0554, 'THR')
S = AminoAcid('S', 11, False, -0.18, 0, 0, -0.8, 5.68, 0.0665, 'SER')
N = AminoAcid('N', 12, False, -0.78, 0, 0, -3.5, 5.41, 0.0383, 'ASN')
Q = AminoAcid('Q', 13, False, -0.85, 0, 0, -3.5, 5.65, 0.0378, 'GLN')
D = AminoAcid('D', 14, True, -0.90, 3.9, -1, -3.5, 2.77, 0.0547, 'ASP')
E = AminoAcid('E', 15, True, -0.74, 4.1, -1, -3.5, 3.22, 0.0618, 'GLU')
H = AminoAcid('H', 16, False, -0.40, 6.5, 0, -3.2, 7.59, 0.0219, 'HIS')
R = AminoAcid('R', 17, True, -2.53, 12.5, 1, -4.5, 10.76, 0.0575, 'ARG')
K = AminoAcid('K', 18, True, -1.50, 10.8, 1, -3.9, 9.74, 0.0494, 'LYS')
P = AminoAcid('P', 19, False, 0.12, 0, 0, -1.6, 6.30, 0.0486, 'PRO')
GAP = AminoAcid('-', 20, False, 0, 0, 0, 0, 0, 0, 'NA')

AMINOACIDS = [C, M, F, I, L, V, W, Y, A, G, T, S, N, Q, D, E, H, R, K, P, GAP]

LETTER2AA = {a.letter: a for a in AMINOACIDS}
