# run.py
# -*- coding: utf-8 -*-

import os
import logging
import traceback
import inspect
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, CLIPProcessor

# -----------------------
# Logger
# -----------------------
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# -----------------------
# Optional imports: datasets
# -----------------------
FakeNet_dataset = None
try:
    from FakeNet_dataset import FakeNet_dataset
    logger.info("FakeNet_dataset imported successfully.")
except Exception as e:
    logger.warning(f"Failed to import FakeNet_dataset: {e}")

WeiboDataLoaderClass = None
try:
    from utils.clip_dataloader import bert_data as WeiboDataLoaderClass
    logger.info("utils.clip_dataloader.bert_data imported successfully for Weibo.")
except Exception as e:
    logger.warning(f"Failed to import Weibo dataloader: {e}")

Weibo21DataLoaderClass = None
try:
    from utils.weibo21_clip_dataloader import bert_data as Weibo21DataLoaderClass
    logger.info("utils.weibo21_clip_dataloader.bert_data imported successfully for Weibo21.")
except Exception as e:
    logger.warning(f"Failed to import Weibo21 dataloader: {e}")

# -----------------------
# Optional imports: trainers
# -----------------------
GossipCopTrainer = None
WeiboTrainer = None
Weibo21Trainer = None

try:
    # NOTE:
    # In your uploaded file model/domain_gossipcop.py,
    # the training class name is DOMAINTrainerWeibo rather than Trainer.
    from model.domain_gossipcop import DOMAINTrainerWeibo as GossipCopTrainer
    logger.info("model.domain_gossipcop.DOMAINTrainerWeibo imported successfully.")
except Exception as e:
    logger.warning(f"Failed to import GossipCop trainer: {e}")

try:
    # In your uploaded model/domain_weibo.py, the class name is Trainer.
    from model.domain_weibo import Trainer as WeiboTrainer
    logger.info("model.domain_weibo.Trainer imported successfully.")
except Exception as e:
    logger.warning(f"Failed to import Weibo trainer: {e}")

try:
    # In your uploaded model/domain_weibo21.py, the class name is also Trainer.
    from model.domain_weibo21 import Trainer as Weibo21Trainer
    logger.info("model.domain_weibo21.Trainer imported successfully.")
except Exception as e:
    logger.warning(f"Failed to import Weibo21 trainer: {e}")

# -----------------------
# Global tokenizer / processor for GossipCop
# -----------------------
bert_tokenizer_gossipcop = None
clip_processor_gossipcop = None


# -----------------------
# Collate function for GossipCop
# -----------------------
def collate_fn_gossipcop(batch):
    """
    Filter out None samples.
    Stack tensor fields automatically.
    Keep non-tensor fields as Python lists.
    """
    original_len = len(batch)
    batch = [item for item in batch if item is not None]

    if original_len > 0 and not batch:
        logger.warning("Collate(gossipcop): all samples are None, returning None.")
        return None
    if not batch:
        logger.warning("Collate(gossipcop): empty batch after filtering, returning None.")
        return None

    keys = batch[0].keys()
    collated = {}

    for key in keys:
        values = [item[key] for item in batch]
        if all(isinstance(v, torch.Tensor) for v in values):
            try:
                collated[key] = torch.stack(values, dim=0)
            except RuntimeError as e:
                logger.error(f"Failed to stack tensor field '{key}': {e}")
                for i, v in enumerate(values):
                    logger.error(f"  item #{i} shape: {getattr(v, 'shape', None)}")
                return None
        else:
            collated[key] = values

    return collated


class Run:
    """
    Unified runner for:
        - gossipcop
        - weibo
        - weibo21

    Expected config keys:
        Common:
            dataset, model_name, lr, batchsize, emb_dim, max_len,
            num_workers, early_stop, epoch, save_param_dir,
            weight_decay, model_params

        GossipCop:
            gossipcop_data_dir
            gossipcop_reasoning_csv_path
            bert_model_path_gossipcop
            clip_model_path_gossipcop

        Weibo:
            weibo_data_dir
            bert_model_path_weibo
            vocab_file
            clip_model_path_weibo   (recommended)

        Weibo21:
            weibo21_data_dir
            bert_model_path_weibo
            vocab_file
            clip_model_path_weibo21 (recommended, fallback supported)
    """

    def __init__(self, config):
        self.config = config
        self.use_cuda = config.get("use_cuda", torch.cuda.is_available())

        # Common settings
        self.dataset = config["dataset"]
        self.model_name = config["model_name"]
        self.lr = config["lr"]
        self.batchsize = config["batchsize"]
        self.emb_dim = config["emb_dim"]
        self.max_len = config["max_len"]
        self.num_workers = config.get("num_workers", 0)
        self.early_stop = config["early_stop"]
        self.epoch = config["epoch"]
        self.save_param_dir = config["save_param_dir"]

        logger.info(f"Run initialized. dataset={self.dataset}, model_name={self.model_name}")
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}, using CUDA: {self.use_cuda}")

        if self.dataset == "gossipcop":
            self._init_gossipcop()
        elif self.dataset == "weibo":
            self._init_weibo()
        elif self.dataset == "weibo21":
            self._init_weibo21()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")

    # -----------------------
    # Dataset-specific init
    # -----------------------
    def _init_gossipcop(self):
        self.root_path = self.config.get("gossipcop_data_dir")
        self.reasoning_csv_path = self.config.get("gossipcop_reasoning_csv_path")
        self.bert_model_path = self.config.get("bert_model_path_gossipcop")
        self.clip_model_path = self.config.get("clip_model_path_gossipcop")

        self.category_dict = {"gossip": 0}
        self.early_stop_metric_key = self.config.get("early_stop_metric", "F1")

        required = {
            "gossipcop_data_dir": self.root_path,
            "gossipcop_reasoning_csv_path": self.reasoning_csv_path,
            "bert_model_path_gossipcop": self.bert_model_path,
            "clip_model_path_gossipcop": self.clip_model_path,
        }
        for k, v in required.items():
            if not v:
                raise ValueError(f"GossipCop missing config: {k}")

        global bert_tokenizer_gossipcop, clip_processor_gossipcop
        try:
            logger.info(f"Loading GossipCop BERT tokenizer from: {self.bert_model_path}")
            bert_tokenizer_gossipcop = BertTokenizer.from_pretrained(self.bert_model_path)
        except Exception as e:
            logger.error(f"Failed to load GossipCop BERT tokenizer: {e}")

        try:
            logger.info(f"Loading GossipCop CLIP processor from: {self.clip_model_path}")
            clip_processor_gossipcop = CLIPProcessor.from_pretrained(self.clip_model_path)
        except Exception as e:
            logger.error(f"Failed to load GossipCop CLIP processor: {e}")

        if bert_tokenizer_gossipcop is None or clip_processor_gossipcop is None:
            raise RuntimeError("GossipCop tokenizer / processor could not be loaded.")

    def _init_weibo(self):
        self.root_path = self.config.get("weibo_data_dir")
        if not self.root_path:
            raise ValueError("Weibo missing config: weibo_data_dir")

        self.train_path = os.path.join(self.root_path, "train_origin.csv")
        self.val_path = os.path.join(self.root_path, "val_origin.csv")
        self.test_path = os.path.join(self.root_path, "test_origin.csv")

        self.category_dict = {
            "经济": 0, "健康": 1, "军事": 2, "科学": 3, "政治": 4,
            "国际": 5, "教育": 6, "娱乐": 7, "社会": 8
        }
        self.bert_model_path = self.config.get("bert_model_path_weibo")
        self.vocab_file = self.config.get("vocab_file")

        # domain_weibo.py needs clip_path_or_name
        self.clip_model_path = (
            self.config.get("clip_model_path_weibo")
            or self.config.get("clip_model_path")
            or self.config.get("clip_model_path_gossipcop")
        )

        self.early_stop_metric_key = self.config.get("early_stop_metric", "acc")

        if not self.bert_model_path:
            raise ValueError("Weibo missing config: bert_model_path_weibo")
        if not self.vocab_file:
            raise ValueError("Weibo missing config: vocab_file")
        if not self.clip_model_path:
            raise ValueError(
                "Weibo missing clip model path. Please provide clip_model_path_weibo or clip_model_path."
            )

    def _init_weibo21(self):
        self.root_path = self.config.get("weibo21_data_dir")
        if not self.root_path:
            raise ValueError("Weibo21 missing config: weibo21_data_dir")

        self.train_path = os.path.join(self.root_path, "train_datasets.xlsx")
        self.val_path = os.path.join(self.root_path, "val_datasets.xlsx")
        self.test_path = os.path.join(self.root_path, "test_datasets.xlsx")

        self.category_dict = {
            "科技": 0, "军事": 1, "教育考试": 2, "灾难事故": 3, "政治": 4,
            "医药健康": 5, "财经商业": 6, "文体娱乐": 7, "社会生活": 8
        }
        self.bert_model_path = self.config.get("bert_model_path_weibo")
        self.vocab_file = self.config.get("vocab_file")

        # domain_weibo21.py also needs clip_path_or_name
        self.clip_model_path = (
            self.config.get("clip_model_path_weibo21")
            or self.config.get("clip_model_path_weibo")
            or self.config.get("clip_model_path")
            or self.config.get("clip_model_path_gossipcop")
        )

        self.early_stop_metric_key = self.config.get("early_stop_metric", "acc")

        if not self.bert_model_path:
            raise ValueError("Weibo21 missing config: bert_model_path_weibo")
        if not self.vocab_file:
            raise ValueError("Weibo21 missing config: vocab_file")
        if not self.clip_model_path:
            raise ValueError(
                "Weibo21 missing clip model path. Please provide clip_model_path_weibo21 or clip_model_path."
            )

    # -----------------------
    # Dataloaders
    # -----------------------
    def get_dataloader(self):
        logger.info(f"Preparing dataloaders for dataset: {self.dataset}")

        train_loader, val_loader, test_loader = None, None, None

        if self.dataset == "gossipcop":
            if FakeNet_dataset is None:
                raise ImportError("FakeNet_dataset is not available.")

            img_size = 224
            clip_max_len = 77

            logger.info("Initializing GossipCop train dataset...")
            train_dataset = FakeNet_dataset(
                root_path=self.root_path,
                reasoning_csv_path=self.reasoning_csv_path,
                bert_tokenizer_instance=bert_tokenizer_gossipcop,
                clip_processor_instance=clip_processor_gossipcop,
                dataset_name="gossip",
                image_size=img_size,
                is_train=True,
                bert_max_len=self.max_len,
                clip_max_len=clip_max_len,
            )

            logger.info("Initializing GossipCop validation dataset...")
            val_dataset = FakeNet_dataset(
                root_path=self.root_path,
                reasoning_csv_path=self.reasoning_csv_path,
                bert_tokenizer_instance=bert_tokenizer_gossipcop,
                clip_processor_instance=clip_processor_gossipcop,
                dataset_name="gossip",
                image_size=img_size,
                is_train=False,
                bert_max_len=self.max_len,
                clip_max_len=clip_max_len,
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batchsize,
                shuffle=True,
                collate_fn=collate_fn_gossipcop,
                num_workers=self.num_workers,
                drop_last=True,
                pin_memory=self.use_cuda,
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batchsize,
                shuffle=False,
                collate_fn=collate_fn_gossipcop,
                num_workers=self.num_workers,
                drop_last=False,
                pin_memory=self.use_cuda,
            )

            test_loader = val_loader

        elif self.dataset == "weibo":
            if WeiboDataLoaderClass is None:
                raise ImportError("utils.clip_dataloader.bert_data is not available for Weibo.")

            loader = WeiboDataLoaderClass(
                max_len=self.max_len,
                batch_size=self.batchsize,
                vocab_file=self.vocab_file,
                category_dict=self.category_dict,
                num_workers=self.num_workers,
                clip_model_name="ViT-B-16",
                clip_download_root="./",
            )

            train_loader = loader.load_data(
                self.train_path,
                os.path.join(self.root_path, "train_loader.pkl"),
                os.path.join(self.root_path, "train_clip_loader.pkl"),
                True,
            )
            val_loader = loader.load_data(
                self.val_path,
                os.path.join(self.root_path, "val_loader.pkl"),
                os.path.join(self.root_path, "val_clip_loader.pkl"),
                False,
            )
            test_loader = loader.load_data(
                self.test_path,
                os.path.join(self.root_path, "test_loader.pkl"),
                os.path.join(self.root_path, "test_clip_loader.pkl"),
                False,
            )

        elif self.dataset == "weibo21":
            if Weibo21DataLoaderClass is None:
                raise ImportError("utils.weibo21_clip_dataloader.bert_data is not available for Weibo21.")

            loader = Weibo21DataLoaderClass(
                max_len=self.max_len,
                batch_size=self.batchsize,
                vocab_file=self.vocab_file,
                category_dict=self.category_dict,
                num_workers=self.num_workers,
            )

            train_loader = loader.load_data(
                self.train_path,
                os.path.join(self.root_path, "train_loader.pkl"),
                os.path.join(self.root_path, "train_clip_loader.pkl"),
                True,
            )
            val_loader = loader.load_data(
                self.val_path,
                os.path.join(self.root_path, "val_loader.pkl"),
                os.path.join(self.root_path, "val_clip_loader.pkl"),
                False,
            )
            test_loader = loader.load_data(
                self.test_path,
                os.path.join(self.root_path, "test_loader.pkl"),
                os.path.join(self.root_path, "test_clip_loader.pkl"),
                False,
            )

        else:
            raise ValueError(f"Unsupported dataset: {self.dataset}")

        if train_loader is None or val_loader is None or test_loader is None:
            missing = []
            if train_loader is None:
                missing.append("train_loader")
            if val_loader is None:
                missing.append("val_loader")
            if test_loader is None:
                missing.append("test_loader")
            raise RuntimeError(
                f"Dataloader initialization failed for {self.dataset}: {', '.join(missing)}"
            )

        logger.info(
            f"Dataloaders ready. train={len(train_loader) if hasattr(train_loader, '__len__') else 'N/A'}, "
            f"val={len(val_loader) if hasattr(val_loader, '__len__') else 'N/A'}, "
            f"test={len(test_loader) if hasattr(test_loader, '__len__') else 'N/A'}"
        )

        return train_loader, val_loader, test_loader

    # -----------------------
    # Trainer builder
    # -----------------------
    def build_trainer(self, train_loader, val_loader, test_loader, save_dir):
        model_params = self.config.get("model_params", {})
        mlp_cfg = model_params.get("mlp", {})
        mlp_dims = mlp_cfg.get("dims", [384])
        dropout = mlp_cfg.get("dropout", 0.2)
        weight_decay = self.config.get("weight_decay", 5e-5)

        if self.dataset == "gossipcop":
            if GossipCopTrainer is None:
                raise ImportError("GossipCop trainer has not been imported.")

            trainer = GossipCopTrainer(
                emb_dim=self.emb_dim,
                mlp_dims=mlp_dims,
                bert=self.bert_model_path,
                use_cuda=self.use_cuda,
                lr=self.lr,
                dropout=dropout,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                category_dict=self.category_dict,
                weight_decay=weight_decay,
                save_param_dir=save_dir,
                reasoning_emb_dim=self.config.get("reasoning_emb_dim", 768),
                num_manipulation_classes=self.config.get("num_manipulation_classes", 0),
                distillation_weight=self.config.get("distillation_weight", 0.1),
                lambda_manipulation_predict=self.config.get("lambda_manipulation_predict", 0.1),
                early_stop=self.early_stop,
                epoches=self.epoch,
            )
            return trainer

        if self.dataset == "weibo":
            if WeiboTrainer is None:
                raise ImportError("Weibo trainer has not been imported.")

            trainer = WeiboTrainer(
                emb_dim=self.emb_dim,
                mlp_dims=mlp_dims,
                bert_path_or_name=self.bert_model_path,
                clip_path_or_name=self.clip_model_path,
                use_cuda=self.use_cuda,
                lr=self.lr,
                dropout=dropout,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                category_dict=self.category_dict,
                weight_decay=weight_decay,
                save_param_dir=save_dir,
                distillation_weight=self.config.get("distillation_weight", 0.5),
                early_stop=self.early_stop,
                epoches=self.epoch,
                metric_key_for_early_stop=self.early_stop_metric_key,
            )
            return trainer

        if self.dataset == "weibo21":
            if Weibo21Trainer is None:
                raise ImportError("Weibo21 trainer has not been imported.")

            trainer = Weibo21Trainer(
                emb_dim=self.emb_dim,
                mlp_dims=mlp_dims,
                bert_path_or_name=self.bert_model_path,
                clip_path_or_name=self.clip_model_path,
                use_cuda=self.use_cuda,
                lr=self.lr,
                dropout=dropout,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                category_dict=self.category_dict,
                weight_decay=weight_decay,
                save_param_dir=save_dir,
                distillation_weight=self.config.get("distillation_weight", 0.5),
                early_stop=self.early_stop,
                epoches=self.epoch,
                metric_key_for_early_stop=self.early_stop_metric_key,
            )
            return trainer

        raise ValueError(f"No trainer build logic defined for dataset: {self.dataset}")

    # -----------------------
    # Main
    # -----------------------
    def main(self):
        logger.info(f"Starting run: dataset={self.dataset}, model_name={self.model_name}")

        try:
            train_loader, val_loader, test_loader = self.get_dataloader()
        except Exception:
            logger.error("Failed during dataloader construction:\n" + traceback.format_exc())
            return

        save_dir = os.path.join(self.save_param_dir, f"{self.dataset}_{self.model_name}")
        os.makedirs(save_dir, exist_ok=True)
        logger.info(f"Model checkpoints will be saved to: {save_dir}")

        try:
            trainer = self.build_trainer(train_loader, val_loader, test_loader, save_dir)
        except Exception:
            logger.error("Failed during trainer construction:\n" + traceback.format_exc())
            return

        logger.info("Trainer initialized successfully. Training starts now...")
        try:
            result = trainer.train()
            logger.info(f"Training finished successfully: {self.dataset} - {self.model_name}")
            return result
        except Exception:
            logger.error("Training crashed:\n" + traceback.format_exc())
            return


if __name__ == "__main__":
    logger.info("This file is usually called from main.py via Run(config).main()")