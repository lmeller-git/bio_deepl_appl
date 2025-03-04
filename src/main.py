from argparse import ArgumentParser
from torch import nn
import builtins


def main(args):
    builtins.VERBOSITY = args.verbosity
    builtins.OUT = args.out

    if args.verbosity == 0:
        builtins.print = lambda *args, **kwargs: None

    from src import (
        train,
        TrainParams,
        BasicMLP,
        MLP,
        Siamese,
        dist_plot,
        TriameseNetwork,
        cluster_plot,
        ModelParams,
        make_predictions,
    )

    # print(args)
    # model = Siamese(hidden_dim=512, n_layers=2) #LeakyMLP(768)
    model = MLP(
        ModelParams(act=nn.ReLU)
    )
    if args.mode == "cv":
        params = TrainParams(
            args.data,
            args.out,
            args.epochs,
            args.lr,
            args.batchsize,
            args.folds,
            d_lr=args.delta_lr,
            d_epoch=args.delta_epochs,
            d_batch_size=args.delta_batchsize,
        )
    elif args.mode == "anal":
        dist_plot(args.data + "project_data/mega_train.csv")
        dist_plot(args.data + "project_data/mega_val.csv")
        dist_plot(args.data + "project_data/mega_test.csv")
        cluster_plot(args.data + "project_data/mega_train.csv")
        cluster_plot(args.data + "project_data/mega_val.csv")
        cluster_plot(args.data + "project_data/mega_test.csv")
        return
    elif args.mode == "predict":
        _ = make_predictions(args.wt, args.mutations)
        return
    else:
        params = TrainParams(args.data, args.out, args.epochs, args.lr, args.batchsize)
    train(model, params)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--data", "-d", type=str, default="./data/", help="path to project_data dir"
    )

    parser.add_argument(
        "--verbosity", "-v", type=int, default=2, help="verbosity of the program"
    )

    subparsers = parser.add_subparsers(dest="mode")

    anal_parser = subparsers.add_parser("anal", help="data analysis")

    inference_parser = subparsers.add_parser(
        "predict", help="predict ddG values for all provided mutations"
    )

    inference_parser.add_argument(
        "wt", type=str, help="fasta file containing wt seq, or a nt seq"
    )

    inference_parser.add_argument(
        "mutations",
        type=str,
        nargs="+",
        help="Mutations to check. Can be any of a fasta file, a list of sequences, a list of mutations in format <wt nt><position><mut nt>",
    )

    cv_parser = subparsers.add_parser("cv", help="Enable cross-validation")
    cv_parser.add_argument(
        "--folds", type=int, default=5, help="Number of folds for cross-validation"
    )
    cv_parser.add_argument(
        "--delta_epochs",
        type=int,
        default=0,
        help="Epoch adjustment per model",
    )
    cv_parser.add_argument(
        "--delta_lr",
        type=float,
        default=0.0,
        help="Learning rate adjustment per model",
    )
    cv_parser.add_argument(
        "--delta_batchsize",
        type=int,
        default=0,
        help="Batch size adjustment per model",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2.0e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        default=1024,
        help="Batch size",
    )
    parser.add_argument(
        "--out", "-o", type=str, default="./out/", help="dir to save outputs"
    )
    args = parser.parse_args()
    main(args)
