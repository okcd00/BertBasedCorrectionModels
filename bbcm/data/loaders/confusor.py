"""
@Time   :   2021-08-03 17:38:56
@File   :   confusor.py
@Author :   okcd00
@Email  :   okcd00{at}qq.com
"""

import sys
sys.path.append("../../")

import pickle
from tqdm import tqdm
import random
from bbcm.utils.pinyin_utils import PinyinUtils
from bbcm.utils.confuser_utils import (
    generate_score_matrix,
    edit_distance_filtering,
    refined_edit_distance,
    cosine_similarity,
    bow_similarity_filtering
)


# CONFUSOR_DATA_DIR = '/home/pangchaoxu/'
CONFUSOR_DATA_DIR = '/data/chendian/'
SCORE_MAT_PATH = f'{CONFUSOR_DATA_DIR}/tencent_embedding/score_data/'
EMBEDDING_PATH = f'{CONFUSOR_DATA_DIR}/tencent_embedding/sound_tokens/'
CORPUS_PATH = f'{CONFUSOR_DATA_DIR}/tencent_embedding/pinyin2token.pkl'
SIGHAN_CFS_PATH = '/home/chendian/BBCM/datasets/sighan_confusion.txt'


class Confusor(object):
    def __init__(self, amb_score=0.5, inp_score=0.25, threshold=(0.2, 0.5),
                 cand_pinyin_num=10, weight=0.5, conf_size=10,
                 filter_strategy='bow', mode='sort', debug=False):
        """
        @param amb_score: [0, 1) score of the ambiguous sounds.
        @param inp_score: [0, 1) score of the input errors.
        @param cand_pinyin_num: the number of candidate pinyin sequences.
        @param threshold: the threshold of the cosine similarity filtering.
        @param mode: {'sort', 'random'} the 'sort' mode sorts candidates by weighted scores.
        @param filter_strategy: {'bow', 'ED', 'no'} 'bow' for bow similarity filtering; 'ED' for edit distance filtering.
        @param weight: final_score = -weight * pinyin_score + cosine_similarity.
        @param conf_size: the size of confusion set.
        """
        self.debug = debug
        self.amb_score = amb_score
        self.inp_score = inp_score
        self.threshold = threshold
        self.cand_pinyin_num = cand_pinyin_num
        self.weight = weight
        self.conf_size = conf_size
        self.filter_strategy = filter_strategy
        self.pu = PinyinUtils()

        print("Use {} mode.".format(mode))
        print("Use {} filtering strategy.".format(filter_strategy))
        self.mode = mode
        self.char_confusion_set = {}
        self.word_confusion_set = {}  # a function is better than a dict
        self.load_sighan_confusion_set()
        self.load_word_confusion_set()

        # pinyin2token corpus
        print("Now loading pinyin2token corpus.")
        self.corpus = pickle.load(open(CORPUS_PATH, 'rb'))

        # load and generate the score matrix
        print("Now generating score matrix.")
        self.score_matrix = self.load_score_matrix()

    def load_sighan_confusion_set(self):
        for line in open(SIGHAN_CFS_PATH, 'r'):
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

    def load_embeddings(self, tokens):
        """
        Given a list of tokens, return the dict {token: embedding}.
        """
        starts = []
        for t in tokens:
            start = self.pu.to_pinyin(t)[0]
            if start not in starts:
                starts.append(start)
        print("Load word embeddings.")
        tok2emb = {}
        for start in starts:
            emb_dict = pickle.load(open(EMBEDDING_PATH + start + '.pkl', 'rb'))
            tok2emb.update(emb_dict)
        tok_embeddings = {tok: tok2emb[tok] for tok in tokens}
        return tok_embeddings

    def load_score_matrix(self):
        print("Load and generate the score matrix.")
        amb_data = pickle.load(open(SCORE_MAT_PATH + 'amb_data.pkl', 'rb'))
        inp_data = pickle.load(open(SCORE_MAT_PATH + 'inp_data.pkl', 'rb'))
        self.score_matrix = generate_score_matrix(
            amb_data, self.amb_score, inp_data, self.inp_score)
        return self.score_matrix

    def get_pinyin_sequence(self, token, corpus, cand_pinyin_num, filter_strategy):
        """
        @param corpus: a dict {token_len:{pinyin: [tokens]}}
        @return: The top-down pinyin sequences.
        """
        pinyin = ''.join(self.pu.to_pinyin(token))
        cand_py = {}
        filter_strategy = filter_strategy.lower()
        # print("Edit distance filtering.")
        if filter_strategy == 'ed':
            filtered_py = edit_distance_filtering(pinyin, list(corpus[len(token)].keys()))
        elif filter_strategy == 'bow':
            filtered_py = bow_similarity_filtering(pinyin, list(corpus[len(token)].keys()))
        elif filter_strategy == 'no':
            filtered_py = list(corpus[len(token)].keys())
        else:
            raise ValueError("invalid filtering strategy: {}".format(filter_strategy))
        # print("Refined edit distance filtering.")
        for pyseq in tqdm(filtered_py):
            score = refined_edit_distance(pinyin, pyseq, self.score_matrix)
            cand_py[pyseq] = score
        top_cand = sorted(cand_py.items(), key=lambda x: x[1])
        return top_cand[:cand_pinyin_num]

    def get_confuse_tokens(self, token, corpus, pinyin_scores, threshold, mode, weight, size):
        """
        @param corpus: a dict {token_len:{pinyin: [tokens]}}
        @param weight: final_score = -weight * pinyin_score + cosine_similarity
        """
        cand_p = [p[0] for p in pinyin_scores]
        candpy2score = {p[0]: p[1] for p in pinyin_scores}
        cand_tokens = [token]
        for pin in cand_p:
            cand_tokens.extend(corpus[len(token)][pin])
        tok2emb = self.load_embeddings(cand_tokens)
        filtered_cand_toks = []
        for tok in cand_tokens:
            cos_sim = cosine_similarity(tok2emb[token], tok2emb[tok])
            if threshold[0] <= cos_sim <= threshold[1]:
                filtered_cand_toks.append(tok)
        if self.debug:
            print("{} candidate tokens in total.".format(len(filtered_cand_toks)))
        if mode == 'random':
            random.shuffle(filtered_cand_toks)
            return filtered_cand_toks[:size]
        elif mode == 'sort':
            cand2score = {}
            for tok in filtered_cand_toks:
                cosine_sim = cosine_similarity(tok2emb[token], tok2emb[tok])
                pinyin_score = candpy2score[''.join(self.pu.to_pinyin(tok))]
                final_score = -weight*pinyin_score + cosine_sim
                cand2score[tok] = final_score
            sort_cand = sorted(cand2score.items(), key=lambda x: x[1], reverse=True)
            return [p[0] for p in sort_cand[1:size + 1]]
        else:
            raise ValueError("invalid mode: {}".format(mode))

    def __call__(self, word, context=None, word_position=None):
        cand_pinyin = self.get_pinyin_sequence(word, self.corpus, self.cand_pinyin_num, self.filter_strategy)
        confusion_set = self.get_confuse_tokens(word, self.corpus, cand_pinyin, self.threshold, self.mode,
                                                self.weight, self.conf_size)
        return confusion_set


if __name__ == "__main__":
    conf = Confusor(threshold=(0.1, 0.5), filter_strategy='bow', mode='sort')
    print(conf('其实'))
