import numpy as np
import re
import time
import faiss

# global config
recallx_at_y=[(1,1), (1,10), (1,100), (15,100), (200,5000)] # recall x @ y
K = max(j for i,j in recallx_at_y)
PATH_QUERY = ""
PATH_GROUNDTRUTH = ""
PATH_BASE = ""

parser = argparse.ArgumentParser(description="This program test recall and qps of different ann search index/method with faiss lib.")
parser.add_argument("algo", choices=['ivfpq', 'ivf-indenp-pq'], help="which index algorithm to apply")
parser.add_argument("--nlists", type=int, default=1024, help="the number of clusters of ivf.")
parser.add_argument("--nbits", type=int, default=8, help="the code width of pq/sq.")
parser.add_argument("--pqdim", type=int, default=32, help="the number of pq subspaces.")
args = parser.parse_args()

# dataset path
ds = args.dataset
if ds == "ood400m":
    PATH_QUERY = "/home/cuixk/dataset/newOOD/ood400m/query.i8bin"
    PATH_GROUNDTRUTH = "/home/cuixk/dataset/newOOD/ood400m/gt.u64bin"
    PATH_BASE = "/home/cuixk/dataset/newOOD/ood400m/base.i8bin"
elif ds == "ood600m":
    PATH_QUERY = "/home/cuixk/dataset/newOOD/ood600m/query.i8bin"
    PATH_GROUNDTRUTH = "/home/cuixk/dataset/newOOD/ood600m/gt.u64bin"
    PATH_BASE = "/home/cuixk/dataset/newOOD/ood600m/base.i8bin"
elif ds == "ood10m":
    PATH_QUERY = "/home/cuixk/dataset/newOOD/ood10m/query.i8bin"
    PATH_GROUNDTRUTH = "/home/cuixk/dataset/newOOD/ood10m/gt.u64bin"
    PATH_BASE = "/home/cuixk/dataset/newOOD/ood10m/base.i8bin"
elif ds == "webface25m256d":
    PATH_QUERY = "/home/yangshuo/data/webface25m256d/query.fbin"
    PATH_GROUNDTRUTH = "/home/yangshuo/data/webface25m256d/gt.ibin"
    PATH_BASE = "/home/yangshuo/data/webface25m256d/base.fbin"
elif ds == "sift1m":
    PATH_QUERY = "/home/yangshuo/data/sift1m/query.fbin"
    PATH_GROUNDTRUTH = "/home/yangshuo/data/sift1m/gt.ibin"
    PATH_BASE = "/home/yangshuo/data/sift1m/base.fbin"
else:
    print("dataset path needed.")
    exit()
    
def read_bin(fname):
    # Parse the postfix of the file name with Regular Expression
    match = re.search(r'(\D+)(\d*)bin$', fname.split('.')[-1])
    if not match:
        raise ValueError("Invalid postfix, must end with *bin")
    data_type_prefix = match.group(1)
    bits = match.group(2)

    if bits == '':
        bits = '32'
    bits = int(bits)
    
    if data_type_prefix == 'i':
        dtype = np.dtype(f'int{bits}')
    elif data_type_prefix == 'u':
        dtype = np.dtype(f'uint{bits}')
    elif data_type_prefix == 'f':
        dtype = np.dtype(f'float{bits}')
    else:
        raise ValueError("Invalid data type, must be 'i', 'u', or 'f'")
    
    shape = np.fromfile(fname, dtype=np.uint32, count=2)
    data = np.memmap(fname, dtype=dtype, offset=8, mode='r').reshape(shape)
    return data, shape[0], shape[1]
    
def load_dataset():
    print("\nloading ", ds)
    x_base, n_base, dim_base = read_bin(PATH_BASE)
    x_query, n_query, dim_query = read_bin(PATH_QUERY)
    gt, n_gt, k_gt = read_bin(PATH_GROUNDTRUTH)
    assert  dim_base == dim_query, "dim_base != dim_query"
    assert n_query == n_gt, "n_query != n_gt"
    print(time.strftime("[%H:%M:%S]", time.localtime()), "Have loaded %d base of %s data type with %d dim, %d queries and gt with top-%d." %
        (n_base, x_base.dtype, dim_base, n_query, k_gt)
    )
    return x_base, x_query, gt, int(dim_base)
    
def calc_recall_for_single_query(res_i, gt_i, X, Y):
    k_temp = gt_i.size - np.sum(gt_i == -1)  # Calculate the valid elements in gt_i
    real_x = min(X, k_temp)
    tp = sum(res_i[j] in gt_i[:real_x] for j in range(Y))
    return tp / real_x

def calc_recall(res, gt):
    for x, y in recallx_at_y:
        n_res, k_res = res.shape
        n_gt, k_gt = gt.shape
        assert n_res == n_gt, "n_res and n_gt should be equal"
        assert k_res >= y, "k_res must be at least Y"
        # Use joblib's Parallel and delayed to process each row in parallel
        recall_list = Parallel(n_jobs=-1)(delayed(calc_recall_for_single_query)(res[i], gt[i], x, y) for i in range(n_res))
        recall = np.mean(recall_list)
        print(f"recall{x}@{y}: {recall:.4f}")
        
def searchDetail(index, xq, K, gt):
    print(f"\nsearching {K}-nn with index {args.algo}...")
    t0 = time.time()
    _, I = index.search(xq, K)
    print(time.strftime("[%H:%M:%S]", time.localtime()), f"Done. Search time: {time.time() - t0 :.2f} seconds")
    # calculate recall
    if not args.noRecall:
        print("\nrecall:")
        calc_recall(I, gt)

def search(index, xq, K, gt):
    if args.gpu and K>2048:
        index = faiss.index_gpu_to_cpu(index)
        print("\nK is too large. search on CPU.")
    if args.algo[0:3]=="ivf":
        for np in args.nprobe:
            index.nprobe = np
            print("\nset nprobe =", np)
            searchDetail(index, xq, K, gt)
    else:
        searchDetail(index, xq, K, gt)

if __name__=='__main__':
    xb, xq, gt, dim = load_dataset()
    
    if algo=="ivfpq":
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFPQ(quantizer, dim, args.nlists, args.pqdim, args.nbits)
        search(index, xq, K, gt)
    elif algo=="ivf-indenp-pq":
        pass
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, args.nlists)
