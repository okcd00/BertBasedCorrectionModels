import torch
import random
from copy import deepcopy


class DataCollatorForCsc:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, data):
        ori_texts, cor_texts, wrong_idss = zip(*data)
        encoded_texts = [self.tokenizer.tokenize(t) for t in ori_texts]
        max_len = max([len(t) for t in encoded_texts]) + 2
        det_labels = torch.zeros(len(ori_texts), max_len).long()
        for i, (encoded_text, wrong_ids) in enumerate(zip(encoded_texts, wrong_idss)):
            for idx in wrong_ids:
                margins = []
                for word in encoded_text[:idx]:
                    if word == '[UNK]':
                        break
                    if word.startswith('##'):
                        margins.append(len(word) - 3)
                    else:
                        margins.append(len(word) - 1)
                margin = sum(margins)
                move = 0
                while (abs(move) < margin) or (idx + move >= len(encoded_text)) or encoded_text[idx + move].startswith(
                        '##'):
                    move -= 1
                det_labels[i, idx + move + 1] = 1
        return ori_texts, cor_texts, det_labels


class DynamicDataCollatorForCsc(DataCollatorForCsc):
    def __init__(self, tokenizer, augmentation=False):
        super(DynamicDataCollatorForCsc, self).__init__(tokenizer)
        self.augmentation = augmentation
        self.char_confusion_set = {}
        self.word_confusion_set = {}
        # TODO

    def sample_augment(self, ori_text, cor_text, wrong_ids):
        # change ori_text here
        ori_text_case, cor_text_case, wrong_ids_case = [], cor_text, []
        for o, c, w in zip(ori_text, cor_text, wrong_ids):
            ot = deepcopy(o)
            wr_ids = deepcopy(w)
            for wid in w:
                o_char = o[wid]
                candidates = self.char_confusion_set.get(o_char)
                # TODO: modify words with word_confusion_set
                ot[wid] = random.choice(candidates)
                wr_ids.append(wid)
            ori_text_case.append(ot)
            wrong_ids_case.append(wr_ids)
        return ori_text_case, cor_text_case, wrong_ids_case

    def __call__(self, data):
        if self.augmentation:
            ori_texts, cor_texts, wrong_idss = self.sample_augment(*zip(*data))
        else:
            ori_texts, cor_texts, wrong_idss = zip(*data)

        encoded_texts = [self.tokenizer.tokenize(t) for t in ori_texts]
        max_len = max([len(t) for t in encoded_texts]) + 2
        det_labels = torch.zeros(len(ori_texts), max_len).long()
        for i, (encoded_text, wrong_ids) in enumerate(zip(encoded_texts, wrong_idss)):
            for idx in wrong_ids:
                margins = []
                for word in encoded_text[:idx]:
                    if word == '[UNK]':
                        break
                    if word.startswith('##'):
                        margins.append(len(word) - 3)
                    else:
                        margins.append(len(word) - 1)
                margin = sum(margins)
                move = 0
                while (abs(move) < margin) or (idx + move >= len(encoded_text)) \
                        or encoded_text[idx + move].startswith('##'):
                    move -= 1
                det_labels[i, idx + move + 1] = 1
        return ori_texts, cor_texts, det_labels