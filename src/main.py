from modelling import train, TrainParams, BasicMLP
from argparse import ArgumentParser


def main(args):
    model = BasicMLP(768)
    params = TrainParams()
    train(model, params)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "data", type=str, default="./data/", help="path to project_data dir"
    )
    args = parser.parse_args()
    main(args)
