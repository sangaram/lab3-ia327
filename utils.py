import re

def clean(yagoEntity):
    if yagoEntity.startswith('"'):
        return yagoEntity[1:-1]
    find = yagoEntity.find(':')
    if find >= 0:
        yagoEntity = yagoEntity[yagoEntity.find(':') + 1:]
        return '<' + yagoEntity + '>'
    return yagoEntity

def run_evaluation():
    results_path = "results-dev.tsv"
    with open(results_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    predictions = {}

    for line in lines:
        split_line = line.strip().split('\t')
        if len(split_line) != 3:
            pass
        subject = clean(split_line[0])
        relation = split_line[1]
        obj = clean(split_line[2])
        predictions[(subject, obj)] = relation

    with open("silver-train.tsv", "r", encoding="utf-8") as f:
        lines = f.readlines()
    ground_truth = {}

    for line in lines:
        split_line = line.strip().split('\t')
        if len(split_line) != 4:
            pass
        subject = clean(split_line[0])
        relation = split_line[-1]
        obj = clean(split_line[1])
        ground_truth[(subject, obj)] = relation

    true_pos = 0
    false_pos = 0
    false_neg = 0
    for subject, obj in ground_truth.keys():
        try:
            pred_relation = predictions[(subject, obj)]
        except KeyError:
            continue
        gold_relation = ground_truth[(subject, obj)]
        if gold_relation != "no_rel":
            if pred_relation == gold_relation:
                true_pos += 1
            elif pred_relation == "no_rel" :
                false_neg += 1
            else:
                false_pos += 1

    if true_pos + false_pos != 0:
        precision = float(true_pos) / (true_pos + false_pos)
    else:
        precision = 0.0

    if true_pos + false_neg != 0:
        recall = float(true_pos) / (true_pos + false_neg + false_pos)
    else:
        recall = 0.0

    beta = 2

    if precision + recall != 0.0:
        f05 = (1 + beta * beta) * precision * recall / (beta * beta * precision + recall)
    else:
        f05 = 0.0

    print()
    print("Scores on the dev set (scaled from 0 to 100)")
    print("Precision", precision * 100)
    print("Recall", recall * 100)
    print("F-2 Score", f05 * 100)