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
        load_model,
        cross_validate,
        load_df,
        plot_mut_dist,
        make_structure_pred,
    )

    # print(args)
    # model = Siamese(hidden_dim=512, n_layers=2) #LeakyMLP(768)
    model = TriameseNetwork(
        ModelParams(hidden_dim=256, n_layers=2, act=nn.ReLU),
        ModelParams(hidden_dim=512, n_layers=3, act=nn.ReLU),
        ModelParams(n_layers=2, hidden_dim=512),
    )  # MLP(ModelParams(act=nn.ReLU))
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
    elif args.mode == "analysis":
        plot_mut_dist(args.data)
        dist_plot(args.data)
        # cluster_plot(args.data + "project_data/mega_train.csv")
        # cluster_plot(args.data + "project_data/mega_val.csv")
        # cluster_plot(args.data + "project_data/mega_test.csv")
        model = load_model(OUT + "best_model.pth")
        train_df, val_df, test_df = load_df(args.data, args.batchsize)
        cross_validate(model, val_df, args.data + "project_data/mega_val.csv")
        cross_validate(model, test_df, args.data + "project_data/mega_test.csv")
        cross_validate(model, train_df, args.data + "project_data/mega_train.csv")
        return
    elif args.mode == "predict":
        if args.prediction_mode == "whole":
            make_structure_pred(args.wt, args.pdb)
            return
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

    anal_parser = subparsers.add_parser("analysis", help="data analysis")

    inference_parser = subparsers.add_parser(
        "predict", help="predict ddG values for all provided mutations"
    )

    inf_parser = inference_parser.add_subparsers(dest="prediction_mode")

    single_parser = inf_parser.add_parser(
        "single", help="make a single prediction for all given mutations"
    )

    single_parser.add_argument(
        "wt", type=str, help="fasta file containing wt seq, or a nt seq"
    )

    single_parser.add_argument(
        "mutations",
        type=str,
        nargs="+",
        help="Mutations to check. Can be any of a fasta file, a list of sequences, a list of mutations in format <wt nt><position><mut nt>",
    )

    struct_parser = inf_parser.add_parser(
        "whole",
        help="do a prediction on each residue and show averaged ddG changes per Residue",
    )
    struct_parser.add_argument(
        "wt", type=str, help="fasta file containing wt seq, or a nt seq"
    )
    struct_parser.add_argument("pdb", type=str, help="pdb file")

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
