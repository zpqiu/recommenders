from reco_utils.recommender.deeprec.deeprec_utils import cal_metric

lines = open('examples/test_mind/results/npa-valid.txt', 'r').readlines()
group_labels, group_preds = [],[]
metrics = ["group_auc", "mean_mrr", "ndcg@5;10"]

for l in lines:
    r = l.strip().split("\t")
    group_labels.append([int(float(x)) for x in r[0].split(",")])
    group_preds.append([float(x) for x in r[1].split(",")])

res = cal_metric(group_labels, group_preds, metrics)
print(res)
