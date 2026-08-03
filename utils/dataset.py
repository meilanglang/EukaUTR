from heapq import merge
from typing import List, Any, Optional, Collection, Tuple, Dict
from pathlib import Path
#from codonMamba.benchmark.data.zero_shot.processing_RNAGym import p
from numpy.ma import remainder
import torch
import sys
from utils.evo.ffindex import MSAFFindex
from utils.evo.tokenization import Vocab
from utils.evo.typed import PathLike
from utils.evo.dataset import CollatableVocabDataset, NPZDataset, JsonDataset, A3MDataset, PickleDataset
from utils.evo.codon import codon_to_aa_pretrain as codon_to_aa
from utils.evo.codon import codon_to_aa_finetune
from utils.evo.tensor import collate_tensors
import utils.rna_esm
import re
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pickle
from Bio import SeqIO
import pandas as pd
import math 
import h5py

class UTRESM2Dataset(CollatableVocabDataset):
    def __init__(
            self,
            #data_path: PathLike,
            config,
            vocab: Vocab,
            data_split: Optional[Collection[str]] = None,
    ):
        super().__init__(vocab)
        #data_path = Path(data_path)
        self.downstream_type = config.downstream_type
        data_path = config.seqfile
        df = pd.read_csv(data_path) #Sequence,Value,Dataset,Split
        split_df = df[df["Split"] == data_split] #train or test
        self.seqs = split_df[["Sequence","Value"]].values.tolist()
        self.sizes = np.array([len(item[0]) for item in self.seqs])

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, index):
        rna_id = index
        seq = self.seqs[index][0]
        #print("seq:",str(seq).decode())
        seq = seq.upper()
        seq = re.sub(r"[T]", "U", seq)
        seq = re.sub(r"[RYKMSWBDHVN~]|\.|\*", "X", seq)
        #print(seq)
        #print(self.vocab)
        if self.downstream_type == "finetune":
            #one_hot_feat = torch.from_numpy(self.one_hot_encode(seq))
            tokens = torch.from_numpy(self.vocab.encode(seq))
            #print(tokens)
            label = torch.from_numpy(np.array(self.seqs[index][1])).float()
            return tokens,label,len(seq)

    def one_hot_encode(self, sequences):
        sequences_arry = np.array(list(sequences)).reshape(-1, 1)
        lable = np.array(list('ACGU')).reshape(-1, 1)
        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit(lable)
        seq_encode = enc.transform(sequences_arry).toarray()
        return seq_encode



class ESMEmbeddingLoader:
    def __init__(self, h5_path):
        self.h5_path = h5_path
        self.hf = h5py.File(self.h5_path, 'r')

    def get_embedding(self, rna_id):
        try:
            return torch.from_numpy(self.hf[rna_id][:])
        except KeyError:
            print(f"Warning: {rna_id} not found in embeddings")
            return None
        except Exception as e:
            print(f"Error accessing embeddings: {e}")
            return None


class IDTokenDataset(CollatableVocabDataset):
    def __init__(self, vocab, data_path):
        super().__init__(vocab)
        self.data = list(SeqIO.parse(data_path, "fasta"))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        if not 0 <= index < len(self):
            raise IndexError(index)
        item = self.data[index]
        rna_id = item.id
        seq = item.seq
        token = torch.from_numpy(self.vocab.encode_single_sequence(seq))
        return rna_id, token



    def outer_concatenation(self, matrix_A, matrix_B):
        "l1xn--> l1xl1x2n"
        matrix_A = matrix_A
        matrix_B = matrix_Bclass SSDataset(CollatableVocabDataset):
    def __init__(
            self,
            data_path: PathLike,
            msa_path: PathLike,
            label_path: PathLike,
            vocab: Vocab,
            split_files: Optional[Collection[str]] = None,
            max_seqs_per_msa: Optional[int] = 64,
            sample_method: str = "fast",
    ):
        super().__init__(vocab)

        data_path = Path(data_path)
        msa_path = Path(msa_path)
        self.rnaids = split_files
        self.a3m_data = A3MDataset(
            data_path / msa_path,
            split_files=split_files,
            max_seqs_per_msa=max_seqs_per_msa,
            sample_method=sample_method  # "fast", "best"
        )
        self.npz_data = NPZDataset(
            data_path / label_path, split_files=split_files, lazy=True
        )
        assert len(self.a3m_data) == len(self.npz_data)

    def get(self, key: str):
        msa = self.a3m_data.get(key)
        tokens = torch.from_numpy(self.vocab.encode(msa))
        missing_nt_index = torch.from_numpy(self.npz_data[key]['missing_nt_index'])
        contacts = torch.from_numpy(self.npz_data[key]['olabel'])
        return tokens, contacts, missing_nt_index

    def __len__(self) -> int:
        return len(self.a3m_data)

    def __getitem__(self, index):
        rnaid = self.rnaids[index]
        msa = self.a3m_data[index]
        one_hot_feat = self.one_hot_encode(msa)
        pairwise_onehot = torch.from_numpy(self.outer_concatenation(one_hot_feat, one_hot_feat))
        tokens = torch.from_numpy(self.vocab.encode(msa))
        contacts = torch.from_numpy(self.npz_data[index]['olabel']).float()
        missing_nt_index = torch.from_numpy(self.npz_data[index]['missing_nt_index']).type(torch.long)
        return rnaid, pairwise_onehot, tokens, contacts, missing_nt_index

    def one_hot_encode(self, sequences):
        sequences_arry = np.array(list(sequences)).reshape(-1, 1)
        lable = np.array(list('ACGU')).reshape(-1, 1)
        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit(lable)
        seq_encode = enc.transform(sequences_arry).toarray()
        return seq_encode

    def outer_concatenation(self, matrix_A, matrix_B):
        "l1xn--> l1xl1x2n"
        matrix_A = matrix_A
        matrix_B = matrix_B
        matrix_C = np.zeros((matrix_A.shape[0], matrix_B.shape[0], matrix_A.shape[1] + matrix_B.shape[1]))
        for v1_dx, v1 in enumerate(matrix_A):
            for v2_dx, v2 in enumerate(matrix_B):
                v12 = np.hstack([v1, v2])
                matrix_C[v1_dx, v2_dx] = v12
        return matrix_C

    def collater(
            self, batch: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        rnaid, tokens, contacts, missing_nt_index = tuple(zip(*batch))
        src_tokens = collate_tensors(tokens, constant_value=self.vocab.pad_idx)
        targets = collate_tensors(contacts, constant_value=-1, dtype=torch.long)
        src_lengths = torch.tensor(
            [contact.size(1) for contact in contacts], dtype=torch.long
        )

        result = {
            'rna_id': rnaid,
            'src_tokens': src_tokens,
            'tgt': targets,
            'tgt_lengths': src_lengths,
            'missing_nt_index': missing_nt_index,
        }
        return result





