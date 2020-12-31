import os

ROOT_PATH = "test_mind/results"

imp_id = 1

with open(os.path.join(ROOT_PATH, "nrms-test-ep4.txt"), "w") as fw:
    for pno in range(20):
        lines = open(os.path.join(ROOT_PATH, "nrms-test-prediction.p{}.txt".format(pno)), "r").readlines()
        for l in lines:
            rank = l.strip().split()[1]
            fw.write("{} {}\n".format(imp_id, rank))
            imp_id += 1
