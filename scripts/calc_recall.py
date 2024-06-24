import numpy as np
from joblib import Parallel, delayed

N_QUERY = 100
K = 5
recall_x_at_y = [(5,5)]

def read_bin(fname, shape, dtype):
    data = np.memmap(fname, dtype=dtype, mode='r').reshape(shape)
    return data

def calc_recall_for_single_query(res_i, gt_i, X, Y):
    k_temp = gt_i.size - np.sum(gt_i == -1)  # Calculate the valid elements in gt_i
    real_x = min(X, k_temp)
    tp = sum(res_i[j] in gt_i[:real_x] for j in range(Y))
    return tp / real_x

def calc_recall(res, gt):
    for x, y in recall_x_at_y:
        n_res, k_res = res.shape
        n_gt, k_gt = gt.shape
        assert n_res == n_gt, "n_res and n_gt should be equal"
        assert k_res >= y, "k_res must be at least Y"
        # Use joblib's Parallel and delayed to process each row in parallel
        recall_list = Parallel(n_jobs=-1)(delayed(calc_recall_for_single_query)(res[i], gt[i], x, y) for i in range(n_res))
        recall = np.mean(recall_list)
        print(f"recall{x}@{y}: {recall:.4f}")
      
res = read_bin("/home/algo/xdu/opPQ/pqsearch/2_verify_op/run/out/result_files/output_0.bin", (N_QUERY, K), np.int32)
gt_I = read_bin("/home/algo/xdu/normal_cpu/myPQ/data/int8/gt_I.bin", (N_QUERY, K), np.int32)
calc_recall(res, gt_I)

# for x,y in recall_x_at_y:
#     total_recall = 0
#     for i in range(N_QUERY):
#         for j in range(y):
#             for k in range(x):
#                 if res[i][j] == gt_I[i][k]:
#                     total_recall += 1
