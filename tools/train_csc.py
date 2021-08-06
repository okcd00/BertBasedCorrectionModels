"""
@Time   :   2021-01-21 11:47:09
@File   :   train_csc.py
@Author :   Abtion
@Email  :   abtion{at}outlook.com
"""
import sys
sys.path.append('..')

from transformers import BertTokenizer
from pytorch_lightning.callbacks import ModelCheckpoint

from bases import args_parse, train, dynamic_train
from bbcm.data.build import make_loaders, get_dynamic_loader
from bbcm.data.loaders import get_csc_loader
from bbcm.data.loaders.collator import DataCollatorForCsc, DynamicDataCollatorForCsc
from bbcm.data.processors.csc import preproc, preproc_cd
from bbcm.modeling.csc import SoftMaskedBertModel
from bbcm.modeling.csc.modeling_bert4csc import BertForCsc
from bbcm.utils import get_abs_path
import os


def main():
    cfg = args_parse("csc/train_bert4csc.yml")

    # 如果不存在训练文件则先处理数据
    # if not os.path.exists(get_abs_path(cfg.DATASETS.TRAIN)):
    #     preproc()
    preproc_cd()
    tokenizer = BertTokenizer.from_pretrained(cfg.MODEL.BERT_CKPT)
    if cfg.MODEL.NAME in ["bert4csc", "macbert4csc"]:
        model = BertForCsc(cfg, tokenizer)
    else:
        model = SoftMaskedBertModel(cfg, tokenizer)

    if len(cfg.MODEL.WEIGHTS) > 0:
        ckpt_path = get_abs_path(cfg.OUTPUT_DIR, cfg.MODEL.WEIGHTS)
        model.load_from_checkpoint(ckpt_path, cfg=cfg, tokenizer=tokenizer)

    loaders = make_loaders(cfg, get_csc_loader,
                           _collate_fn=DataCollatorForCsc(tokenizer=tokenizer))
    # loaders = make_dynamic_loaders(cfg, get_csc_loader, _collate_fn=None)
    ckpt_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=get_abs_path(cfg.OUTPUT_DIR),
        filename='{epoch:02d}_{val_loss:.5f}',
        save_top_k=1,
        mode='min'
    )
    train(cfg, model, loaders, ckpt_callback)


def dynamic_main(fixed=False):
    cfg = args_parse("csc/train_bert4csc.yml")
    fixed = cfg.TASK.get('FIXED', False)

    # 如果不存在训练文件则先处理数据
    # if not os.path.exists(get_abs_path(cfg.DATASETS.TRAIN)):
    #     preproc()
    # preproc_cd()
    tokenizer = BertTokenizer.from_pretrained(cfg.MODEL.BERT_CKPT)
    if cfg.MODEL.NAME in ["bert4csc", "macbert4csc"]:
        model = BertForCsc(cfg, tokenizer)
    else:
        model = SoftMaskedBertModel(cfg, tokenizer)

    if len(cfg.MODEL.WEIGHTS) > 0:
        ckpt_path = get_abs_path(cfg.OUTPUT_DIR, cfg.MODEL.WEIGHTS)
        model.load_from_checkpoint(ckpt_path, cfg=cfg, tokenizer=tokenizer)

    loaders = make_loaders(cfg, get_csc_loader,
                           _collate_fn=DataCollatorForCsc(tokenizer=tokenizer))
    # loaders = make_dynamic_loaders(cfg, get_csc_loader, _collate_fn=None)
    ckpt_callback = ModelCheckpoint(
        monitor=None,
        dirpath=get_abs_path(cfg.OUTPUT_DIR),
        filename='{epoch:02d}_{val_loss:.5f}',
        save_top_k=-1,
    )
    dynamic_train(cfg, model, loaders, ckpt_callback, fixed=fixed)


if __name__ == '__main__':
    dynamic_main()
