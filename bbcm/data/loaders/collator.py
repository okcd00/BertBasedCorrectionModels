import random
from copy import deepcopy

import torch
from bbcm.data.loaders.confusor import Confusor
from bbcm.utils.text_utils import is_chinese_char


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
                while (abs(move) < margin) or (idx + move >= len(encoded_text)) \
                        or encoded_text[idx + move].startswith('##'):
                    move -= 1
                det_labels[i, idx + move + 1] = 1
        return ori_texts, cor_texts, det_labels


class DynamicDataCollatorForCsc(DataCollatorForCsc):
    def __init__(self, tokenizer, augmentation=False):
        super(DynamicDataCollatorForCsc, self).__init__(tokenizer)
        self.first_epoch = True  # False 
        self.augmentation = augmentation
        self.confusor = Confusor(cand_pinyin_num=10)

    def change_words(self, word, correct_word=None, sentence=None):
        if len(word) == 1:
            candidates = self.confusor(word)
            candidates += self.confusor.char_confusion_set.get(word, [])
            # can = deepcopy(candidates)
            return random.choice(candidates) if candidates else word
        else:
            candidates = self.confusor(word)
        if correct_word and correct_word in candidates:
            candidates.remove(correct_word)
        return word

    def random_wrong_ids(self, ct, wrong_id, word_offsets=None):
        ot = deepcopy(ct)
        text_len = ot.__len__()
        candidate_position = [i for i in range(text_len) if is_chinese_char(ct[i])]
        n_faulty_position = len(wrong_id)
        wrong_ids = sorted(random.sample(candidate_position,
                                         max(1, n_faulty_position)))
        if word_offsets:
            wrong_ids = set(wrong_ids)
            for wid in wrong_ids:
                wrong_ids.update(word_offsets[wid])
        return ot, sorted(wrong_ids)

    def generate_word_offsets(self, ct):
        word_offsets = {}
        word_indexes = [0]
        words = self.confusor.pu.segmentation(ct)
        for w in words:
            wc = [_i for _i in range(word_indexes[-1], word_indexes[-1] + len(w))]
            word_offsets.update({_i: wc for _i in wc})
            word_indexes.append(len(w) + word_indexes[-1])
        return word_offsets

    def sample_augment(self, ori_text, cor_text, wrong_ids, random_pos=True, word_level=True):
        ori_text_case, cor_text_case, wrong_ids_case = [], cor_text, []
        # ori_text, cor_text, wrong_ids are all lists
        for o, c, w in zip(ori_text, cor_text, wrong_ids):
            ot, wr_ids = deepcopy(o), deepcopy(w)
            word_offsets = self.generate_word_offsets(c)
            if random_pos:  # change another position to augment
                ot, wr_ids = self.random_wrong_ids(
                    ct=c, wrong_id=wr_ids, word_offsets=None)
            done_wid_list = []
            for wid in wr_ids:
                if wid in done_wid_list:
                    continue
                _word = o[wid]
                _correct_word = c[wid]
                if word_level and random.random() > 0.5:
                    word_len = len(word_offsets[wid])
                    for wid_in_word in word_offsets[wid]:
                        if wid_in_word in wr_ids:
                            done_wid_list.append(wid_in_word)

                cw = self.change_words(  # HERE
                    word=_word, correct_word=_correct_word, sentence=c)
                # change ori_text here
                ot = f"{ot[:wid]}{cw}{ot[wid+1:]}"
                # if cw != c[wid]: current_wids.append(wid)
            current_wids = [_id for _id in
                            range(len(c)) if c[_id] != ot[_id]]
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
