# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# fmt: off
proteinseq_toks = {
    'toks': ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X', 'B', 'U', 'Z', 'O', '.', '-']
}

rnaseq_toks = {
    'toks': ['A', 'G', 'C', 'U', 'X']    #加了‘-’和‘X’,去掉’T‘
}
# fmt: on
#- in msa
codon_toks = {'toks':['AAA', 'AAC', 'AAG', 'AAU', 'ACA', 'ACC', 'ACG', 'ACU',
                      'AGA', 'AGC', 'AGG', 'AGU', 'AUA', 'AUC', 'AUG', 'AUU',
                      'CAA', 'CAC', 'CAG', 'CAU', 'CCA', 'CCC', 'CCG', 'CCU',
                      'CGA', 'CGC', 'CGG', 'CGU', 'CUA', 'CUC', 'CUG', 'CUU',
                      'GAA', 'GAC', 'GAG', 'GAU', 'GCA', 'GCC', 'GCG', 'GCU',
                      'GGA', 'GGC', 'GGG', 'GGU', 'GUA', 'GUC', 'GUG', 'GUU',
                      'UAA', 'UAC', 'UAG', 'UAU', 'UCA', 'UCC', 'UCG', 'UCU',
                      'UGA', 'UGC', 'UGG', 'UGU', 'UUA', 'UUC', 'UUG', 'UUU',
                      "NNN","---"]
}

codonseq_toks = {'toks': ['AAA', 'AAU', 'AAC', 'AAG', 'AUA', 'AUU', 'AUC', 'AUG',
                          'ACA', 'ACU', 'ACC', 'ACG', 'AGA', 'AGU', 'AGC', 'AGG',
                          'UAA', 'UAU', 'UAC', 'UAG', 'UUA', 'UUU', 'UUC', 'UUG',
                          'UCA', 'UCU', 'UCC', 'UCG', 'UGA', 'UGU', 'UGC', 'UGG',
                          'CAA', 'CAU', 'CAC', 'CAG', 'CUA', 'CUU', 'CUC', 'CUG',
                          'CCA', 'CCU', 'CCC', 'CCG', 'CGA', 'CGU', 'CGC', 'CGG',
                          'GAA', 'GAU', 'GAC', 'GAG', 'GUA', 'GUU', 'GUC', 'GUG',
                          'GCA', 'GCU', 'GCC', 'GCG', 'GGA', 'GGU', 'GGC', 'GGG']
                 }