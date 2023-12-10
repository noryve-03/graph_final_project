import argparse

args = argparse.ArgumentParser()
args.add_argument("--graph_path", default="cora.npz")
args.add_argument("--architecture", default=[64])
args.add_argument("--collapse_regularization", default=1)
args.add_argument("--dropout_rate", default=0)
args.add_argument("--n_clusters", default=16)
args.add_argument("--n_epochs", default=1000)
args.add_argument("--learning_rate", default=0.01)

args = args.parse_args()