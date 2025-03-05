# Run once to install the ESM-1b model: https://github.com/facebookresearch/esm

import scipy.spatial.distance

import esm, esm.scripts, esm.scripts.extract
import torch
import shutil

# https://www.uniprot.org/uniprotkb/P04637/entry


def get_emb(seq: str, muts: list[str]) -> tuple[torch.Tensor, list[torch.Tensor]]:
    with open("/tmp/seq.fasta", "w") as f:
        print(f">wt", file=f)

        print(seq, file=f)
        for mut in muts:
            aa_pos = int(mut[1:-1])

            aa_ref = mut[0]

            aa_alt = mut[-1]

            # print(aa_pos, aa_ref, aa_alt)

            mut_seq = seq[: aa_pos - 1] + aa_alt + seq[aa_pos:]

            # print(mut)
            assert seq[aa_pos - 1] == aa_ref

            assert mut_seq[aa_pos - 1] == aa_alt
            print(f">{mut}", file=f)

            print(mut_seq, file=f)

    parser = esm.scripts.extract.create_parser()

    args = parser.parse_args(
        ["esm1_t6_43M_UR50S", "/tmp/seq.fasta", "embeddings", "--include", "mean"]
    )

    esm.scripts.extract.run(args)
    wt_emb = torch.load("embeddings/wt.pt", weights_only=True)["mean_representations"][
        6
    ]
    embs = [
        torch.load("embeddings/" + p + ".pt", weights_only=True)[
            "mean_representations"
        ][6]
        for p in muts
    ]
    lbls = [
        torch.load("embeddings/" + p + ".pt", weights_only=True)["label"] for p in muts
    ]
    shutil.rmtree("embeddings")
    return wt_emb, embs, lbls


def main():
    wt_seq = "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD"

    print(len(wt_seq), "residues in sequence")

    mutations = [
        # TP53 mutation known to destabilize the protein: https://www.pnas.org/doi/10.1073/pnas.0805326105
        "Y220C",
        # Two ClinVar mutations classified as 'benign': https://www.ncbi.nlm.nih.gov/clinvar/?term=Li-Fraumeni+syndrome
        "E298S",
        "Q354K",
    ]

    print(len(mutations), "mutations")
    with open("sequences.fasta", "w") as fh:
        print(f">wt", file=fh)

        print(wt_seq, file=fh)

        for mut in mutations:
            aa_pos = int(mut[1:-1])

            aa_ref = mut[0]

            aa_alt = mut[-1]

            # print(aa_pos, aa_ref, aa_alt)

            mut_seq = wt_seq[: aa_pos - 1] + aa_alt + wt_seq[aa_pos:]

            assert wt_seq[aa_pos - 1] == aa_ref

            assert mut_seq[aa_pos - 1] == aa_alt

            print(f">{mut}", file=fh)

            print(mut_seq, file=fh)

    parser = esm.scripts.extract.create_parser()

    args = parser.parse_args(
        ["esm1_t6_43M_UR50S", "sequences.fasta", "embeddings", "--include", "mean"]
    )

    esm.scripts.extract.run(args)

    # Check shape of arbitrary embedding

    wt_emb = torch.load("embeddings/wt.pt")["mean_representations"][6]

    Y220C_emb = torch.load("embeddings/Y220C.pt")["mean_representations"][6]

    E298S_emb = torch.load("embeddings/E298S.pt")["mean_representations"][6]

    Q354K_emb = torch.load("embeddings/Q354K.pt")["mean_representations"][6]
    print(scipy.spatial.distance.cosine(Y220C_emb, wt_emb))  # pathogenic

    print(scipy.spatial.distance.cosine(E298S_emb, wt_emb))  # benign

    print(scipy.spatial.distance.cosine(Q354K_emb, wt_emb))  # benign


if __name__ == "__main__":
    main()
