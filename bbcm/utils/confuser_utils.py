from tqdm import tqdm
import string
import copy
import numpy as np

zimu2inds = {z: i for i, z in enumerate(list(string.ascii_lowercase))}
zimu2inds['0'] = 26


def zimu2ind(zimu):
    """
    '0' for the start of the sequence. Only applied in del_matrix.
    """
    return zimu2inds[zimu]


def edit_distance(str1, str2):
    """
    Given two sequences, return the edit distance normalized by the max length.
    """
    matrix = [[i + j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]
    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if (str1[i - 1] == str2[j - 1]):
                d = 0
            else:
                d = 1
            matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + d)
            # return matrix
    return matrix[len(str1)][len(str2)] / max([len(str1), len(str2)])


def apply_edit_distance(target, corpus):
    """
    Given a target pinyin, and a pinyin list, return the sorted candidates.
    """
    cand_edit_dist = {}
    for cand_py in tqdm(corpus):
        cand_edit_dist[cand_py] = edit_distance(cand_py, target)
    sort_cand = sorted(cand_edit_dist.items(), key=lambda x:x[1])

    return sort_cand


def edit_distance_filtering(pinyin, pinyin_corpus, cand_num=10000):
    sort_cand = apply_edit_distance(pinyin, pinyin_corpus)
    return [p[0] for p in sort_cand[:cand_num]]


def generate_score_matrix(amb_data, amb_score, inp_data, inp_score):
    """
    Generate score matrices from pkl files.
    :param amb_data:
    :param amb_score:
    :param inp_data:
    :param inp_score:
    :return:
    """
    def apply_mat(target_mat, mat_data, score):
        for firz, dellist in mat_data.items():
            for secz in dellist:
                i = zimu2ind(firz)
                j = zimu2ind(secz)
                target_mat[i][j] -= score
        return target_mat
    del_matrix = [[1 for _ in range(27)] for _ in range(27)]
    rep_matrix = copy.deepcopy(del_matrix)
    for i in range(27):
        for j in range(27):
            if i == j or i == 26 or j == 26:
                rep_matrix[i][j] = 0
    del_matrix = apply_mat(del_matrix, amb_data['del_mat'], amb_score)
    rep_matrix = apply_mat(rep_matrix, amb_data['rep_mat'], amb_score)
    rep_matrix = apply_mat(rep_matrix, inp_data, inp_score)
    return del_matrix, rep_matrix


def refined_edit_distance(str1, str2, score_matrix):
    """
    Given two sequences, return the refined edit distance normalized by the max length.
    """
    del_matrix, rep_matrix = score_matrix
    matrix = [[i + j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]
    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            ind_i1 = zimu2ind(str1[i - 1])
            ind_j1 = zimu2ind(str2[j - 1])
            rep_score = rep_matrix[ind_i1][ind_j1]
            pstr1 = '0' if i == 1 else str1[i - 2]
            pstr2 = '0' if j == 1 else str2[j - 2]
            # 删除a_i
            del_score = del_matrix[ind_i1][zimu2ind(pstr1)]

            # 在a后插入b_j
            ins_score = del_matrix[ind_j1][zimu2ind(pstr2)]

            matrix[i][j] = min(matrix[i - 1][j] + del_score, matrix[i][j - 1] + ins_score,
                               matrix[i - 1][j - 1] + rep_score)
            # return matrix
    return matrix[len(str1)][len(str2)] / max([len(str1), len(str2)])


def cosine_similarity(v1, v2):
    norm = np.linalg.norm(v1)*np.linalg.norm(v2)
    if norm:
        return np.dot(v1, v2)/norm
    else:
        return 0

