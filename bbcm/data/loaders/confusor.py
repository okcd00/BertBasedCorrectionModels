"""
@Time   :   2021-08-03 17:38:56
@File   :   confusor.py
@Author :   okcd00
@Email  :   okcd00{at}qq.com
"""


class Confusor(object):
    def __init__(self):
        self.char_confusion_set = {}
        self.word_confusion_set = {}  # a function is better than a dict
        self.load_sighan_confusion_set()
        self.load_word_confusion_set()

    def load_sighan_confusion_set(self):
        sighan_cf_path = '/home/chendian/BBCM/datasets/sighan_confusion.txt'  # on C14
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

    def __call__(self, word, context=None, word_position=None):
        confusion_set = []
        return confusion_set


if __name__ == "__main__":
    pass
