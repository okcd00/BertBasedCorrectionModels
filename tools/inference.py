"""
@Time   :   2021-02-05 15:33:55
@File   :   inference.py
@Author :   Abtion
@Email  :   abtion{at}outlook.com
"""
import sys
import argparse
import os

import torch
from transformers import BertTokenizer

from tools.bases import args_parse

sys.path.append('..')

from bbcm.modeling.csc import BertForCsc, SoftMaskedBertModel
from bbcm.utils import get_abs_path


def parse_args():
    parser = argparse.ArgumentParser(description="bbcm")
    parser.add_argument(
        "--config_file", default="csc/train_bert4csc.yml", help="config file", type=str
    )
    parser.add_argument(
        "--ckpt_fn", default="epoch=2-val_loss=0.02.ckpt", help="checkpoint file name", type=str
    )
    parser.add_argument("--texts", default=["马上要过年了，提前祝大家心年快乐！"], nargs=argparse.REMAINDER)
    parser.add_argument("--text_file", default='')

    args = parser.parse_args()
    return args


def load_model_directly(ckpt_file, config_file):
    # Example:
    # ckpt_fn = 'SoftMaskedBert/epoch=02-val_loss=0.02904.ckpt' (find in checkpoints)
    # config_file = 'csc/train_SoftMaskedBert.yml' (find in configs)
    
    from bbcm.config import cfg
    cp = get_abs_path('checkpoints', ckpt_file)
    cfg.merge_from_file(get_abs_path('configs', config_file))
    tokenizer = BertTokenizer.from_pretrained(cfg.MODEL.BERT_CKPT)

    if cfg.MODEL.NAME in ['bert4csc', 'macbert4csc']:
        model = BertForCsc.load_from_checkpoint(cp,
                                                cfg=cfg,
                                                tokenizer=tokenizer)
    else:
        model = SoftMaskedBertModel.load_from_checkpoint(cp,
                                                         cfg=cfg,
                                                         tokenizer=tokenizer)
    model.eval()
    model.to(cfg.MODEL.DEVICE)
    return model


def load_model(args):
    from bbcm.config import cfg
    cfg.merge_from_file(get_abs_path('configs', args.config_file))
    tokenizer = BertTokenizer.from_pretrained(cfg.MODEL.BERT_CKPT)
    file_dir = get_abs_path("checkpoints", cfg.MODEL.NAME)
    if cfg.MODEL.NAME in ['bert4csc', 'macbert4csc']:
        model = BertForCsc.load_from_checkpoint(os.path.join(file_dir, args.ckpt_fn),
                                                cfg=cfg,
                                                tokenizer=tokenizer)
    else:
        model = SoftMaskedBertModel.load_from_checkpoint(os.path.join(file_dir, args.ckpt_fn),
                                                         cfg=cfg,
                                                         tokenizer=tokenizer)
    model.eval()
    model.to(cfg.MODEL.DEVICE)

    return model


def evaluate(args, test_loader='test'):
    model = load_model(args)
    model.evaluate_from_loader(loader=test_loader)


def inference(args, detail=False):
    model = load_model(args)
    texts = []
    if os.path.exists(args.text_file):
        with open(args.text_file, 'r', encoding='utf-8') as f:
            for line in f:
                texts.append(line.strip())
    else:
        texts = args.texts
    corrected_texts = model.predict(texts, detail)
    print(corrected_texts)
    return corrected_texts


if __name__ == '__main__':
    arguments = parse_args()
    inference(arguments)
