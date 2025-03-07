from Bio import SeqIO
import re
from src.utils import get_emb, load_model
import torch
import json
import matplotlib
import py3Dmol
import numpy as np


def plot_model(resi: dict[int, float], pdb: str):
    cmap = matplotlib.colormaps["cividis"]

    x_min = min([v for v in resi.values()])  # -1.3
    x_max = max([v for v in resi.values()])  # 0.5

    print(f"{x_min = }")
    print(f"{x_max = }")

    def rescale(dp: float) -> float:
        return (dp - x_min) / (x_max - x_min)

    view = py3Dmol.view()
    with open("pred/pdb.txt", "r") as f:
        path = f.read().strip()

    with open(path, "r") as f:
        while True:
            line = f.readline()
            if line.startswith("ATOM"):
                start = int(line.split()[5])
                break
    print(start)
    view.addModel(open(path, "r").read(), "pdb")
    view.setStyle(
        {
            "cartoon": {
                "colorscheme": {
                    "prop": "resi",
                    "map": {
                        int(k) + start: matplotlib.colors.to_hex(cmap(rescale(v)))
                        for k, v in resi.items()
                    },
                },
                "arrows": True,
            }
        }
    )

    view.zoomTo()
    png = view._make_html()
    with open(OUT + f'protein_plot{pdb.split("/")[-1].split(".")[0]}.html', "w") as f:
        f.write(png)
    print(
        f"image generated and written to {OUT + f'protein_plot{pdb.split("/")[-1].split(".")[0]}.html'}"
    )


def make_predictions(wt: str, muts: list[str], model_p: str) -> list[float]:
    wt = get_sequence(wt)

    muts = get_muts(muts, wt)
    wt_emb, embs, _lbls = get_emb(wt, muts)
    model = load_model(model_p)
    model.eval()
    preds = []
    wt_emb = wt_emb.unsqueeze(0)
    for i, (emb, mut) in enumerate(zip(embs, muts)):
        emb = emb.unsqueeze(0)
        p = model(wt_emb, emb).squeeze()
        print(f"ddG of mutation {mut}: {p:.3f}")
        preds.append(p)

    return preds


def make_structure_pred(wt: str, pdb: str, model_p: str, metric: str = "average"):
    wt = get_sequence(wt)
    # hydrophobic = ("A", "V", "I", "L", "M", "F", "W", "Y")
    # hydrophilic = ("D", "R", "H", "K", "E", "Q", "N", "C", "T", "S")
    stabilizing = ("M", "A", "L", "E", "K")
    amino_acids = [
        "A",
        "R",
        "N",
        "D",
        "C",
        "E",
        "Q",
        "G",
        "H",
        "I",
        "L",
        "K",
        "M",
        "F",
        "P",
        "S",
        "T",
        "W",
        "Y",
        "V",
        "U",
        "O",
    ]
    muts = []
    preds = []
    for i, aa in enumerate(wt):
        preds.append([])
        """
        if aa not in stabilizing:
            continue
        """
        for aa2 in amino_acids:
            if aa2 == aa:
                continue
            muts.append(f"{aa}{i + 1}{aa2}")

        """
        if aa in hydrophobic:
            for aa2 in hydrophilic:
                muts.append(f"{aa}{i + 1}{aa2}")
        elif aa in hydrophilic:
            for aa2 in hydrophobic:
                muts.append(f"{aa}{i + 1}{aa2}")
        """
    wt_emb, embs, _ = get_emb(wt, muts)
    wt_emb = wt_emb.unsqueeze(0)
    model = load_model(model_p)
    model.eval()
    for i, (emb, mut) in enumerate(zip(embs, muts)):
        emb = emb.unsqueeze(0)
        pred = model(wt_emb, emb).squeeze().detach().cpu()
        idx = int(mut[1:-1])
        preds[idx - 1].append(pred)

    preds = {
        i: float(sum(p) / len(p)) if len(p) > 0 else 0.0 for i, p in enumerate(preds)
    }
    print(list(zip(preds.values(), wt)))
    plot_model(preds, pdb)
    return


def get_sequence(s: str) -> str:
    if "fasta" in s or "fa" in s:
        return next(SeqIO.parse(s, "fasta")).seq
    return s


def get_muts(s: list[str], s2: str) -> list[str]:
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
