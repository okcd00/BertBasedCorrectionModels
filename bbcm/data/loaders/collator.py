import random
from copy import deepcopy

import torch
from tqdm import tqdm
from bbcm.data.datasets.csc import *
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
        self.min_wrong_positions = 1
        self.min_wrong_positions_rate = 0.0
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
        candidate_position = [i for i in range(text_len)
                              if is_chinese_char(ord(ct[i]))]
        n_faulty_position = len(wrong_id)
        wrong_ids = sorted(random.sample(candidate_position,
                                         max(n_faulty_position,
                                             self.min_wrong_positions,
                                             int(self.min_wrong_positions_rate * text_len))))
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

    def sample_augment_single(self, ot, ct, wrong_id, random_pos=True, word_level=True):
        o_text, c_text, wr_ids = deepcopy(ot), ct, deepcopy(wrong_id)
        word_offsets = self.generate_word_offsets(c_text)
        if random_pos:  # change another position to augment
            o_text, wr_ids = self.random_wrong_ids(
                ct=c_text, wrong_id=wr_ids, word_offsets=None)
        done_wid_list = []  # done wid in the same word.
        for wid in wr_ids:
            if wid in done_wid_list:
                continue
            word_ids = [wid]
            if word_level:
                if random.random() > 0.5:
                    continue  # char or word
                word_ids = sorted(word_offsets[wid])
                for wid_in_word in word_ids:
                    if wid_in_word in wr_ids:
                        done_wid_list.append(wid_in_word)
            _word = ''.join([o_text[_i] for _i in word_ids])
            _correct_word = ''.join([c_text[_i] for _i in word_ids])
            cw = self.change_words(
                word=_word, correct_word=_correct_word, sentence=c_text)
            # change ori_text here
            o_text = f"{o_text[:word_ids[0]]}{cw}{o_text[word_ids[-1]+1:]}"
            # if cw != c_text[wid]: current_wids.append(wid)
        current_wids = [_id for _id in
                        range(len(c_text)) if c_text[_id] != o_text[_id]]
        return o_text, c_text, current_wids

    def sample_augment(self, ori_text, cor_text, wrong_ids, random_pos=True, word_level=True):
        ori_text_case, cor_text_case, wrong_ids_case = [], cor_text, []
        # ori_text, cor_text, wrong_ids are all lists
        for o, c_text, w in zip(ori_text, cor_text, wrong_ids):
            o_text, _, current_wids = self.sample_augment_single(
                ot=o, ct=c_text, wrong_id=w,
                random_pos=random_pos, word_level=word_level)
            ori_text_case.append(o_text)
            wrong_ids_case.append(current_wids)
        return ori_text_case, cor_text_case, wrong_ids_case

    def samples(self):
        return [sample for s_idx, sample in enumerate(self)]

    def load_csc_dataset(self, csc_data_path, csc_data_type='json'):
        csc_origin_data = []
        if os.path.exists(csc_data_path):
            # load from pure text lines
            dataset_type = PureTextDataset
            if 'json' in csc_data_type:
                # csc_origin_data = json.load(open(csc_data_path, 'r'))
                dataset_type = CscDataset
            csc_origin_data = dataset_type(csc_data_path)
        return csc_origin_data

    def generate_csc_augmented_samples(self, csc_origin_data, csc_data_path=None,
                                       csc_data_type='json', random_pos=True):
        """

        :param csc_origin_data: a dataset from {PureTextDataset, CscDataset}
        :param csc_data_path:
        :param random_pos:
        :return:
        """
        if csc_data_path and (csc_origin_data is None):
            csc_origin_data = self.load_csc_dataset(
                csc_data_path=csc_data_path, csc_data_type=csc_data_type)
        augmented_samples = []
        for s_idx, sample in enumerate(csc_origin_data):
            o, c, w = sample
            o_text, c_text, current_wids = self.sample_augment_single(
                ot=o, ct=c, wrong_id=w, random_pos=random_pos)
            augmented_samples.append({
                # 'id': sample.get('id'),
                'id': f"{s_idx}",
                'original_text': o_text,
                'wrong_ids': current_wids,
                'correct_text': c_text,
            })
        return augmented_samples

    def generate_csc_augmented_samples_from_text_dir(self, csc_data_path, output_dir=None):
        """
        generate from a too-large pure-text corpus.
        :param text_path_pattern:
        :return:
        """
        if output_dir is None:
            output_dir = '/data/sharedata/findoc_csc_samples_210808'
        csc_origin_data = self.load_csc_dataset(
            csc_data_path=csc_data_path, csc_data_type='text')
        augmented_samples = []
        # n_samples = csc_origin_data.__len__()
        last_file = ""
        for s_idx, sample in tqdm(enumerate(csc_origin_data)):
            # o = sample['original_text']
            # c = sample['correct_text']
            # w = sample['wrong_ids']
            o, c, w = sample
            # print(o, c, w)
            o_text, c_text, current_wids = self.sample_augment_single(
                ot=o, ct=c, wrong_id=w)
            if csc_origin_data.current_file_index != last_file:
                out_file = f"{output_dir}/findoc_{csc_origin_data.current_file_index:06}_.json"
                # if os.path.exists(out_file):  continue  # no overwrite
                if out_file:
                    dump_json(augmented_samples, out_file)
                augmented_samples = []
                last_file = f"{csc_origin_data.current_file_index}"
            augmented_samples.append({
                'id': f"{csc_origin_data.current_file_index}_{s_idx}",
                'original_text': o_text,
                'wrong_ids': current_wids,
                'correct_text': c_text,
            })
        else:
            out_file = f"{output_dir}/findoc_{csc_origin_data.current_file_index:06}_.json"
            # if os.path.exists(out_file):  continue  # no overwrite
            if out_file:
                dump_json(augmented_samples, out_file)


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
