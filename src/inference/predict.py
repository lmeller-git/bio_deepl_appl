from Bio import SeqIO
import re
from src.utils import get_emb, load_model
import torch


def make_predictions(wt: str, muts: list[str]) -> list[float]:
    wt = get_sequence(wt)

    muts = get_muts(muts, wt)
    wt_emb, embs = get_emb(wt, muts)
    model = load_model(OUT + "best_model.pth")
    model.eval()
    preds = []
    wt_emb = wt_emb.unsqueeze(0)
    for i, (emb, mut) in enumerate(zip(embs, muts)):
        emb = emb.unsqueeze(0)
        p = model(wt_emb, emb).squeeze()
        print(f"ddG of mutation {mut}: {p:.3f}")
        preds.append(p)

    return preds


def get_sequence(s: str) -> str:
    if "fasta" in s or "fa" in s:
        return SeqIO.parse(s, "fasta")[0]
    return s


def get_muts(s: list[str], s2: str) -> str:
    muts = []
    for seq in s:
        if "fasta" in seq or "fa" in seq:
            for rec in SeqIO.parse(seq, "fasta"):
                muts.append(mut_from_seq(rec, s2))
        elif bool(re.search(r"\d", seq)):
            muts.append(seq)
        else:
            muts.append(mut_from_seq(seq, s2))
    return muts


def mut_from_seq(s1: str, s2: str) -> str:
    for i, (a1, a2) in enumerate(zip(s1, s2)):
        if a1 == a2:
            continue
        return a2 + str(i + 1) + a1
    return ""
