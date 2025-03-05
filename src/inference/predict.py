from Bio import SeqIO
import re
from src.utils import get_emb, load_model
import torch


import __main__

__main__.pymol_argv = ["pymol", "-qc"]  # Quiet mode, no GUI
import pymol
from pymol import cmd
import numpy as np


def color_by_values(pdb_file, value_dict, palette="rainbow"):
    """Color residues by value dictionary using PyMOL API."""
    cmd.load(pdb_file)

    # Normalize values
    values = np.array(list(value_dict.values()))
    min_val, max_val = values.min(), values.max()
    for resi, val in value_dict.items():
        # normalized = (val - min_val) / (max_val - min_val)
        color_name = f"color_{resi}"

        # Gradient from blue (low) to red (high)
        color_rgb = [min_val, 0, max_val]
        cmd.set_color(color_name, color_rgb)
        cmd.color(color_name, f"resi {resi}")

    cmd.show("cartoon")
    cmd.orient()


def make_predictions(wt: str, muts: list[str]) -> list[float]:
    wt = get_sequence(wt)

    muts = get_muts(muts, wt)
    wt_emb, embs, _lbls = get_emb(wt, muts)
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


def make_structure_pred(wt: str, pdb: str, metric: str = "average"):
    wt = get_sequence(wt)
    hydrophobic = ("A", "V", "I", "L", "M", "F", "W", "Y")
    hydrophilic = ("D", "R", "H", "K", "E", "Q", "N", "C", "T", "S")

    muts = []
    preds = []
    for i, aa in enumerate(wt):
        preds.append([])
        if aa in hydrophobic:
            for aa2 in hydrophilic:
                muts.append(f"{aa}{i + 1}{aa2}")
        elif aa in hydrophilic:
            for aa2 in hydrophobic:
                muts.append(f"{aa}{i + 1}{aa2}")
    # print(muts)
    wt_emb, embs, _ = get_emb(wt, muts)
    wt_emb = wt_emb.unsqueeze(0)
    model = load_model(OUT + "best_model.pth")
    model.eval()
    for i, (emb, mut) in enumerate(zip(embs, muts)):
        emb = emb.unsqueeze(0)
        pred = model(wt_emb, emb).squeeze()
        idx = int(mut[1:-1])
        preds[idx].append(pred)

    preds = {i: (sum(p) / len(p)) if len(p) > 0 else 0.0 for i, p in enumerate(preds)}
    pymol.finish_launching()

    color_by_values(pdb, preds)
    cmd.png("colored_protein.png", 1920, 1080, dpi=300)
    cmd.quit()


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
