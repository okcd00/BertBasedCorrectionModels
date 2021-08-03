"""
@Time   :   2021-01-21 11:24:00
@File   :   csc.py
@Author :   Abtion
@Email  :   abtion{at}outlook.com
"""
from torch.utils.data import Dataset
from bbcm.utils import load_json
from bisect import bisect_left, bisect_right
from glob import glob


class CscDataset(Dataset):
    def __init__(self, fp):
        self.data = load_json(fp)
        print(f"Loaded {self.data.__len__()} samples from {fp}.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]['original_text'], self.data[index]['correct_text'], self.data[index]['wrong_ids']


class PureTextDataset(Dataset):
    def __init__(self, fp):
        self.file_list = sorted(glob(f"{fp}/*.txt"))
        self.file_sample_count = []
        self.file_offset = [0]
        self.sample_counts = self.count_samples()
        print(f"Loaded {self.file_list.__len__()} files from {fp}.")

    @staticmethod
    def read_text_file(path):
        return [line.strip() for line in open(path, 'r')]

    def count_samples(self):
        for file_name in self.file_list:
            samples = self.read_text_file(file_name)
            s_len = len(samples)
            self.file_sample_count.append(s_len)
            self.file_offset.append(self.file_offset[-1] + s_len)
        return sum(self.file_sample_count)

    def load_from_dir(self, dir_path):
        self.__init__(dir_path)

    @staticmethod
    def binary_search_right(a, x):
        # binary_search_for_file_index
        i = bisect_left(a, x)
        if i:
            return i
        else:
            return 0

    def __len__(self):
        return self.sample_counts

    def __getitem__(self, index):
        # for a large text corpus, shuffle is not recommended.
        file_index = self.binary_search_right(self.file_offset, index) - 1
        if file_index == self.file_list.__len__():
            raise ValueError(f"Invalid index {file_index} with offset {index}")
        file_path = self.file_list[file_index]
        samples = self.read_text_file(file_path)
        target_text = samples[index-self.file_offset[file_index]]
        # return self.data[index]['original_text'], self.data[index]['correct_text'], self.data[index]['wrong_ids']
        return target_text, target_text, []
