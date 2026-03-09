# main.py
# Usage:
#   python main.py --dataset gossipcop
#   python main.py --dataset weibo21
#   python main.py --dataset weibo

import os
import argparse
import logging
import random
import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger("runner")

# ---------------------------
# 1) Presets for three datasets
# ---------------------------
PRESETS = {
    "gossipcop": {
        "model_name": "domain_gossipcop",
        "lr": 3e-4,
        "batchsize": 24,
        "seed": 2024,
        "early_stop": 100,
        "early_stop_metric": "acc",
        "distillation_weight": 0.9,
        "lambda_reasoning_align": None,
    },
    "weibo21": {
        "model_name": "domain_weibo21",
        "lr": 5e-4,
        "batchsize": 64,
        "seed": 3074,
        "early_stop": 100,
        "early_stop_metric": "F1",
        "distillation_weight": 0.5,
        "lambda_reasoning_align": 0.1,
    },
    "weibo": {
        "model_name": "domain_weibo",
        "lr": 2e-4,
        "batchsize": 64,
        "seed": 3074,
        "early_stop": 100,
        "early_stop_metric": "F1",
        "distillation_weight": 0.5,
        "lambda_reasoning_align": 0.1,
    },
}


def pick(user_value, preset_value):
    return preset_value if user_value is None else user_value


# ---------------------------
# 2) Arguments
# ---------------------------
parser = argparse.ArgumentParser(
    description="Unified runner for gossipcop / weibo21 / weibo"
)

parser.add_argument(
    "--dataset",
    choices=["gossipcop", "weibo21", "weibo"],
    required=True,
    help="Dataset to run."
)

# Optional overrides
parser.add_argument("--model_name", default=None)
parser.add_argument("--lr", type=float, default=None)
parser.add_argument("--batchsize", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--early_stop", type=int, default=None)
parser.add_argument("--early_stop_metric", choices=["acc", "F1"], default=None)

# Training and environment
parser.add_argument("--epoch", type=int, default=50)
parser.add_argument("--max_len", type=int, default=197)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--gpu", default="0")
parser.add_argument("--bert_emb_dim", type=int, default=768)
parser.add_argument("--save_param_dir", default="./param_model")
parser.add_argument("--emb_type", default="bert")

# Pretrained model paths
parser.add_argument(
    "--bert_model_path_gossipcop",
    default="./pretrained_model/bert-base-uncased"
)
parser.add_argument(
    "--clip_model_path_gossipcop",
    default="./pretrained_model/clip-vit-base-patch16"
)

parser.add_argument(
    "--bert_model_path_weibo",
    default="./pretrained_model/chinese_roberta_wwm_base_ext_pytorch"
)
parser.add_argument(
    "--bert_vocab_file_weibo",
    default="./pretrained_model/chinese_roberta_wwm_base_ext_pytorch/vocab.txt"
)

# Added for weibo / weibo21 trainers
parser.add_argument(
    "--clip_model_path_weibo",
    default="./pretrained_model/clip-vit-base-patch16",
    help="CLIP model path for Weibo."
)
parser.add_argument(
    "--clip_model_path_weibo21",
    default="./pretrained_model/clip-vit-base-patch16",
    help="CLIP model path for Weibo21."
)

# Data paths
parser.add_argument("--gossipcop_data_dir", default="./gossipcop/")
parser.add_argument(
    "--gossipcop_reasoning_csv_path",
    default="./gossipcop_with_reasoning_encoded.csv"
)

parser.add_argument("--weibo_data_dir", default="./data/")
parser.add_argument("--weibo21_data_dir", default="./Weibo_21/")

# Loss weights
parser.add_argument(
    "--distillation_weight",
    type=float,
    default=None,
    help="Distillation loss weight."
)
parser.add_argument(
    "--lambda_reasoning_align",
    type=float,
    default=None,
    help="Reasoning alignment loss weight."
)

args = parser.parse_args()

# ---------------------------
# 3) Set visible GPU first
# ---------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

try:
    from run import Run
except Exception as e:
    logger.error(f"Failed to import Run from run.py: {e}")
    raise SystemExit(1)

# ---------------------------
# 4) Apply preset + user overrides
# ---------------------------
preset = PRESETS[args.dataset]

current = {
    "dataset": args.dataset,
    "model_name": pick(args.model_name, preset["model_name"]),
    "lr": pick(args.lr, preset["lr"]),
    "batchsize": pick(args.batchsize, preset["batchsize"]),
    "seed": pick(args.seed, preset["seed"]),
    "early_stop": pick(args.early_stop, preset["early_stop"]),
    "early_stop_metric": pick(args.early_stop_metric, preset["early_stop_metric"]),
    "distillation_weight": pick(args.distillation_weight, preset["distillation_weight"]),
    "lambda_reasoning_align": pick(args.lambda_reasoning_align, preset["lambda_reasoning_align"]),
}

# ---------------------------
# 5) Fix randomness
# ---------------------------
random.seed(current["seed"])
np.random.seed(current["seed"])
torch.manual_seed(current["seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(current["seed"])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---------------------------
# 6) Build unified config
# ---------------------------
use_cuda = torch.cuda.is_available()

config = {
    # runtime
    "use_cuda": use_cuda,
    "dataset": current["dataset"],
    "model_name": current["model_name"],

    # data paths
    "gossipcop_data_dir": args.gossipcop_data_dir,
    "gossipcop_reasoning_csv_path": args.gossipcop_reasoning_csv_path,
    "weibo_data_dir": args.weibo_data_dir,
    "weibo21_data_dir": args.weibo21_data_dir,

    # pretrained model paths
    "bert_model_path_gossipcop": args.bert_model_path_gossipcop,
    "clip_model_path_gossipcop": args.clip_model_path_gossipcop,

    "bert_model_path_weibo": args.bert_model_path_weibo,
    "bert_vocab_file_weibo": args.bert_vocab_file_weibo,
    "vocab_file": args.bert_vocab_file_weibo,

    # clip paths for weibo / weibo21
    "clip_model_path_weibo": args.clip_model_path_weibo,
    "clip_model_path_weibo21": args.clip_model_path_weibo21,

    # training settings
    "batchsize": current["batchsize"],
    "max_len": args.max_len,
    "early_stop": current["early_stop"],
    "early_stop_metric": current["early_stop_metric"],
    "num_workers": args.num_workers,
    "emb_type": args.emb_type,
    "weight_decay": 5e-5,
    "model_params": {
        "mlp": {
            "dims": [384],
            "dropout": 0.2
        }
    },
    "emb_dim": args.bert_emb_dim,
    "lr": current["lr"],
    "epoch": args.epoch,
    "seed": current["seed"],
    "save_param_dir": args.save_param_dir,

    # losses
    "distillation_weight": current["distillation_weight"],
    "lambda_reasoning_align": current["lambda_reasoning_align"],
}

# Compatibility fields for older trainer code if needed
if args.dataset in {"weibo", "weibo21"}:
    config["bert"] = args.bert_model_path_weibo

# ---------------------------
# 7) Print final config
# ---------------------------
logger.info("===== Final Config =====")
for k, v in config.items():
    logger.info(f"{k}: {v}")
logger.info("========================")

# ---------------------------
# 8) Launch
# ---------------------------
if __name__ == "__main__":
    if config["use_cuda"]:
        logger.info(f"CUDA is available. Using GPU {args.gpu}")
    else:
        logger.warning("CUDA is not available. Running on CPU.")

    runner = Run(config=config)
    runner.main()
    logger.info("Finished.")