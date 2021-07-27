"""
@Time   :   2021-01-21 12:01:32
@File   :   evaluations.py
@Author :   Abtion
@Email  :   abtion{at}outlook.com
"""


def report_prf(tp, fp, fn, phase, logger):
    # For the detection Precision, Recall and F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
    logger.info(f"The {phase} result is:\n"
                f"precision={precision}, recall={recall} and F1={f1_score}\n"
                f"support: TP={tp}, FP={fp}, FN={fn}")
    return precision, recall, f1_score


def compute_corrector_prf(results, logger, on_detected=True):
    """
    References:
        https://github.com/sunnyqiny/
        Confusionset-guided-Pointer-Networks-for-Chinese-Spelling-Check/blob/master/utils/evaluation_metrics.py
    """
    TP = 0
    FP = 0
    FN = 0
    all_predict_true_index = []
    all_gold_index = []
    for item in results:
        src, tgt, predict, d_predict = item

        gold_index = []
        for i in range(len(list(src))):
            if src[i] == tgt[i]:
                continue
            else:
                gold_index.append(i)
        all_gold_index.append(gold_index)

        predict_index = []
        for i in range(len(list(src))):
            if src[i] == predict[i]:
                continue
            else:
                predict_index.append(i)

        each_true_index = []
        for i in predict_index:
            if i in gold_index:
                TP += 1
                each_true_index.append(i)
            else:
                FP += 1
        for i in gold_index:
            if i in predict_index:
                continue
            else:
                FN += 1
        all_predict_true_index.append(each_true_index)

    # For the detection Precision, Recall and F1
    _, _, detection_f1 = report_prf(TP, FP, FN,
                                    'detection', logger=logger)

    # store FN counts
    n_misreported = int(FN)

    TP = 0
    FP = 0
    FN = 0

    # we only detect those correctly detected location, which is a different from the common metrics since
    # we wanna to see the precision improve by using the confusion set
    for i in range(len(all_predict_true_index)):
        if len(all_predict_true_index[i]) > 0:
            predict_words = []
            for j in all_predict_true_index[i]:
                predict_words.append(results[i][2][j])
                if results[i][1][j] == results[i][2][j]:
                    TP += 1
                else:
                    FP += 1
            for j in all_gold_index[i]:
                if results[i][1][j] in predict_words:
                    continue
                else:
                    FN += 1

    # For the correction Precision, Recall and F1
    _, _, correction_f1 = report_prf(TP, FP, FN,
                                     'correction', logger=logger)
    # common metrics to compare with other baseline methods.
    _, _, correction_cf1 = report_prf(TP, FP, FN + n_misreported,
                                      'correction_common', logger=logger)
    if not on_detected:
        correction_f1 = correction_cf1
    return detection_f1, correction_f1


def compute_sentence_level_prf(results, logger):
    """
    自定义的句级prf，设定需要纠错为正样本，无需纠错为负样本
    :param results:
    :return:
    """

    TP = 0.0
    FP = 0.0
    FN = 0.0
    TN = 0.0
    total_num = len(results)

    for item in results:
        src, tgt, predict, _ = item

        # 负样本
        if src == tgt:
            # 预测也为负
            if tgt == predict:
                TN += 1
            # 预测为正
            else:
                FP += 1
        # 正样本
        else:
            # 预测也为正
            if tgt == predict:
                TP += 1
            # 预测为负
            else:
                FN += 1

    acc = (TP + TN) / total_num
    precision = TP / (TP + FP) if TP > 0 else 0.0
    recall = TP / (TP + FN) if TP > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0

    logger.info(f'Sentence Level: acc:{acc:.6f}, precision:{precision:.6f}, recall:{recall:.6f}, f1:{f1:.6f}')
    return acc, precision, recall, f1
