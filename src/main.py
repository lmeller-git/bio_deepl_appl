from src import train, TrainParams, BasicMLP, MLP, Siamese
from argparse import ArgumentParser
from torch import nn


def main(args):
    print(args)
    model = MLP(
        hidden_dim=(786, 512, 256), act=nn.LeakyReLU
    )  # Siamese(hidden_dim=512, n_layers=2) #LeakyMLP(768)
    if args.mode == "cv":
        params = TrainParams(
            args.data,
            args.epochs,
            args.lr,
            args.batchsize,
            args.folds,
            d_lr=args.delta_lr,
            d_epoch=args.delta_epochs,
            d_batch_size=args.delta_batchsize,
        )
    else:
        params = TrainParams(args.data, args.epochs, args.lr, args.batchsize)
    train(model, params)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--data", "-d", type=str, default="./data/", help="path to project_data dir"
    )
    subparsers = parser.add_subparsers(dest="mode")

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
