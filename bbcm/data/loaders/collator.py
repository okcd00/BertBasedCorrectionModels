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
        self.first_epoch = True
        self.augmentation = augmentation
        self.char_confusion_set = {}
        self.word_confusion_set = {}
        self.load_sighan_confusion_set()

    def load_sighan_confusion_set(self):
        sighan_cf_path = '/home/chendian/BBCM/datasets/sighan_confusion.txt'
        for line in open(sighan_cf_path, 'r'):
            key, val = line.strip().split(':')
            self.char_confusion_set.setdefault(key, [])
            self.char_confusion_set[key].extend([c for c in val])

    def load_word_confusion_set(self):
        # tx_corpus = '/home/chendian/BBCM/datasets/'

        # TODO: pre-processing words with tx embeddings
        # https://ai.tencent.com/ailab/nlp/zh/embedding.html

        # TODO: pre-processing words with
        # https://github.com/fighting41love/funNLP/tree/master/data
        pass

    def change_words(self, word, correct_word=None, sentence=None):
        if len(word) == 1:
            candidates = self.char_confusion_set.get(word, [])
            can = deepcopy(candidates)
            if correct_word and correct_word in can:
                can.remove(correct_word)
            return random.choice(can) if can else word
        # TODO: modify words with word_confusion_set
        return word

    @staticmethod
    def random_wrong_ids(ct, wrong_id):
        ot = deepcopy(ct)
        text_len = ot.__len__()
        n_faulty_position = len(wrong_id)
        wrong_ids = sorted(random.sample(range(text_len), max(1, n_faulty_position)))
        return ot, wrong_ids

    def sample_augment(self, ori_text, cor_text, wrong_ids, random_pos=True):
        ori_text_case, cor_text_case, wrong_ids_case = [], cor_text, []
        # ori_text, cor_text, wrong_ids are all lists
        for o, c, w in zip(ori_text, cor_text, wrong_ids):
            ot, wr_ids = deepcopy(o), deepcopy(w)
            if random_pos:  # change another position to augment
                ot, wr_ids = self.random_wrong_ids(ct=c, wrong_id=wr_ids)
            current_wids = []
            for wid in wr_ids:
                cw = self.change_words(
                    word=o[wid], correct_word=c[wid], sentence=c)
                # change ori_text here
                ot = f"{ot[:wid]}{cw}{ot[wid+1:]}"
                if cw != c[wid]:
                    current_wids.append(wid)
            ori_text_case.append(ot)
            wrong_ids_case.append(current_wids)
        return ori_text_case, cor_text_case, wrong_ids_case

    def samples(self):
        return [sample for s_idx, sample in enumerate(self)]

    def generate_csc_augmented_samples(self, csc_data_path, random_pos=True):
        import json
        csc_origin_data = json.load(open(csc_data_path, 'r'))
        augmented_samples = []
        for sample in csc_origin_data:
            w = sample['wrong_ids']
            c = sample['correct_text']
            o = sample['original_text']
            ot, wr_ids = deepcopy(o), deepcopy(w)
            if random_pos:  # change another position to augment
                ot, wr_ids = self.random_wrong_ids(ct=c, wrong_id=wr_ids)
            current_wids = []
            for wid in wr_ids:
                cw = self.change_words(
                    word=o[wid], correct_word=c[wid], sentence=c)
                ot = f"{ot[:wid]}{cw}{ot[wid+1:]}"
                if cw != c[wid]:
                    current_wids.append(wid)
            augmented_samples.append({
                'id': sample['id'],
                'original_text': ot,
                'wrong_ids': current_wids,
                'correct_text': c,
            })
        return augmented_samples

    def __call__(self, data):
        # return the original samples for the first epoch
        if self.augmentation and not self.first_epoch:
            ori_texts, cor_texts, wrong_idss = self.sample_augment(*zip(*data))
        else:
            ori_texts, cor_texts, wrong_idss = zip(*data)
        self.first_epoch = False

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
