"""
@Time   :   2021-01-21 10:57:33
@File   :   csc_trainer.py
@Author :   Abtion
@Email  :   abtion{at}outlook.com
"""
import torch
import operator
import numpy as np
from transformers import BertTokenizer
from bbcm.utils.evaluations import compute_corrector_prf, compute_sentence_level_prf
from .bases import BaseTrainingEngine


class CscTrainingModel(BaseTrainingEngine):
    """
        用于CSC的BaseModel, 定义了训练及预测步骤
        """

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        # loss weight for cor & det
        self.w = cfg.MODEL.HYPER_PARAMS[0]
        self.tokenizer = BertTokenizer.from_pretrained(cfg.MODEL.BERT_CKPT)
        # threshold for prediction judgment
        self.judge_line = 0.5

    @staticmethod
    def pt2list(_tensor):
        return _tensor.cpu().numpy().tolist()

    def training_step(self, batch, batch_idx):
        ori_text, cor_text, det_labels = batch
        outputs = self.forward(ori_text, cor_text, det_labels)
        loss = self.w * outputs[1] + (1 - self.w) * outputs[0]
        return loss

    def validation_step(self, batch, batch_idx):
        ori_text, cor_text, det_labels = batch

        # 检错loss，纠错loss，检错输出，纠错输出
        outputs = self.forward(ori_text, cor_text, det_labels)
        loss = self.w * outputs[1] + (1 - self.w) * outputs[0]
        det_y_hat = (outputs[2] > self.judge_line).long()
        cor_y_hat = torch.argmax((outputs[3]), dim=-1)
        encoded_x = self.tokenizer(cor_text, padding=True, return_tensors='pt')
        encoded_x.to(self._device)
        cor_y = encoded_x['input_ids']
        cor_y_hat *= encoded_x['attention_mask']

        results = []
        det_acc_labels = []
        cor_acc_labels = []
        for src, tgt, predict, det_predict, det_label in zip(ori_text, cor_y, cor_y_hat, det_y_hat, det_labels):
            _src = self.tokenizer(src, add_special_tokens=False)['input_ids']

            # whether a token from correction is the same as the truth token.
            c_tgt = self.pt2list(tgt[1:len(_src) + 1])
            c_predict = self.pt2list(predict[1:len(_src) + 1])
            cor_acc_arr = [int(_t == _p) for _t, _p in zip(c_tgt, c_predict)]
            cor_acc_labels.append(cor_acc_arr)

            # whether a token from detection is the same as the truth token.
            d_tgt = self.pt2list(det_label[1:len(_src) + 1])
            d_predict = self.pt2list(det_predict[1:len(_src) + 1])
            det_acc_arr = [int(_t == _p) for _t, _p in zip(d_tgt, d_predict)]
            det_acc_labels.append(det_acc_arr)

            # a tuple of lists
            results.append((_src, c_tgt, c_predict, d_predict))

        return loss.cpu().item(), det_acc_labels, cor_acc_labels, results

    def validation_epoch_end(self, outputs):
        det_acc_labels = []
        cor_acc_labels = []
        results = []

        # loss, det_acc_labels, cor_acc_labels, results
        for out in outputs:
            det_acc_labels += out[1]
            cor_acc_labels += out[2]
            results += out[3]
        loss = np.mean([out[0] for out in outputs])
        self.log('val_loss', loss)
        self._logger.info(f'loss: {loss}')
        det_acc = np.array(det_acc_labels).reshape(-1).mean()
        # print(det_acc)
        self._logger.info(f'Detection:\n'
                          f'acc: {det_acc:.4f}')
        cor_acc = np.array(cor_acc_labels).reshape(-1).mean()
        # print(cor_acc)
        self._logger.info(f'Correction:\n'
                          f'acc: {cor_acc:.4f}')
        det_f1, cor_f1 = compute_corrector_prf(results, self._logger, on_detected=True)
        compute_sentence_level_prf(results, self._logger)
        return det_acc, cor_acc, det_f1, cor_f1

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        self._logger.info('\nTest.\n')
        return self.validation_epoch_end(outputs)

    def evaluate_from_loader(self, loader):
        outputs = []
        if isinstance(loader, str):
            from bbcm.data.loaders.collator import DataCollatorForCsc
            _collate_fn = DataCollatorForCsc(tokenizer=self.tokenizer)

            from bbcm.data.build import make_loaders
            trn, dev, tst = make_loaders(cfg=self.cfg,
                                         _collate_fn=_collate_fn)
            loader = {
                'train': trn,
                'dev': dev,
                'valid': dev,
                'test': tst
            }.get(loader, tst)
        for b_idx, batch in enumerate(loader):
            ori_texts, cor_texts, det_labels = batch
            batch = (ori_texts, cor_texts, det_labels.to(self.cfg.MODEL.DEVICE))
            outputs.append(self.validation_step(batch, b_idx))
        return self.validation_epoch_end(outputs)

    def predict(self, texts, detail=False):
        inputs = self.tokenizer(texts, padding=True, return_tensors='pt')
        inputs.to(self.cfg.MODEL.DEVICE)
        with torch.no_grad():
            # 检测输出，纠错输出
            outputs = self.forward(texts)
            y_hat = torch.argmax(outputs[1], dim=-1)
            expand_text_lens = torch.sum(inputs['attention_mask'], dim=-1) - 1
        rst = []
        for t_len, _y_hat in zip(expand_text_lens, y_hat):
            rst.append(self.tokenizer.decode(_y_hat[1:t_len]).replace(' ', ''))
        if detail:
            return outputs, rst
        return rst
