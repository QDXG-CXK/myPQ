import numpy as np
import faiss
import argparse
import time
import re
from numba import jit, prange
from joblib import Parallel, delayed

# global config
recallx_at_y=[(1,1), (1,10), (1,100), (15,100), (200,5000)] # recall x @ y
K = max(j for i,j in recallx_at_y)
PATH_QUERY = ""
PATH_GROUNDTRUTH = ""
PATH_BASE = ""

# arguments parsing
parser = argparse.ArgumentParser(description="This program test recall and qps of different ann search index/method with faiss lib.")
parser.add_argument('-v', '--version', action='version', version='%(prog)s 1.8')
parser.add_argument("algo", choices=['ivf','brute','ivf-hnsw','ivf-prq'], help="which index algorithm to apply")
parser.add_argument("--dataset", choices=['ood400m','ood600m','webface25m256d','sift1m','ood10m'], default="ood10m")
parser.add_argument("--decomposition", choices=['none','pca'], default='none', 
                        help="Whether and Which decomposition method for base vector.") 
parser.add_argument("--quantization", choices=['none','pq','opq'], default='none', 
                        help="Whether and Which quantization method for base vector.")#TODO:opq,PreTransform;opq enable "M%D!=0" situation?
parser.add_argument("--ncomponents", type=int, default=50, help="the number of components after pca transformation.")
parser.add_argument("--nlists", type=int, default=1024, help="the number of clusters of ivf.")
parser.add_argument("--degree", type=int, default=32, help="the degree of graph.")
parser.add_argument("--n0dim", type=int, default=0, help="append n dim with value 0.")
parser.add_argument("--nbits", type=int, default=8, help="the code width of pq/sq.")
parser.add_argument("--pqdim", type=int, default=32, help="the number of pq subspaces.")
parser.add_argument("--nsplits", type=int, default=32, help="the number of residual quantizers.")
parser.add_argument("--nprobe", type=int, nargs='*', default=[10, 128], help="the number of ivf clusters to probe.")
parser.add_argument("--batch_size", type=int, default=32768, help="batch size when adding or quering.")
parser.add_argument("--refineRatio", type=float, default=1.0, help="k0 = k * refine ratio.")#TODO
parser.add_argument("--numSample", type=int, default=100000, help="number of Sample to train.")
parser.add_argument("--noRecall", action='store_true', default=False, help="Whether to calculate recall.")
parser.add_argument("--gpu", action='store_true', default=False, help="Whether to use gpu version.")
parser.add_argument("--outfile", type=str, default=None, help="Path to save the results.")#TODO
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

def write_bin(data, base_fname):
    if not isinstance(data, np.ndarray):
        raise ValueError("data must be a numpy ndarray")
    
    dtype = data.dtype
    if dtype.kind in ('i', 'u', 'f'):
        type_prefix = dtype.kind
        bits = dtype.itemsize * 8
    else:
        raise ValueError("Unsupported data type")

    if bits == 32:
        bits=''
    suffix = f"{type_prefix}{bits}bin"
    fname = f"{base_fname}.{suffix}"
    shape = np.array(data.shape, dtype=np.uint32)
    with open(fname, "wb") as f:
        f.write(shape.tobytes())
        f.write(data.tobytes())

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

def insert_zeros(xb, subDim, N0):
    if(N0>10):
        return insert_zeros_parallel(xb, subDim, N0)

    insert_positions = np.arange(subDim - 1, subDim * (N0 + 1) - 1, subDim)
    for pos in insert_positions:
        xb = np.insert(xb, pos, 0, axis=1)    
    return xb

@jit(nopython=True, parallel=True)
def insert_zeros_parallel(xb, subDim, N0):
    insert_positions = np.arange(subDim - 2, (subDim - 1) * (N0 + 1) - 1, subDim - 1)
    position_map = np.zeros(xb.shape[1] + N0, dtype=np.int32)
    current_col = 0
    insert_idx = 0
    next_insert_pos = insert_positions[insert_idx] if len(insert_positions) > 0 else None    
    for i in range(xb.shape[1]):
        position_map[i] = current_col
        if next_insert_pos is not None and i == next_insert_pos:
            current_col += 1  # move to next position to insert 0
            if insert_idx < len(insert_positions) - 1:
                insert_idx += 1
                next_insert_pos = insert_positions[insert_idx]
            else:
                next_insert_pos = None        
        current_col += 1    
    # create new array
    new_xb = np.zeros((xb.shape[0], xb.shape[1] + N0), dtype=xb.dtype)
    for i in prange(xb.shape[1]):
        new_col_index = position_map[i]
        new_xb[:, new_col_index] = xb[:, i]    
    return new_xb

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

def build(dim, xb, xs):
    algo=args.algo
    index = None
    
    if algo=="brute":
        if args.quantization == "pq":
            index = faiss.IndexPQ(dim, args.pqdim, args.nbits)
        elif args.quantization == "opq":
            dim2 = dim // args.pqdim * args.pqdim
            subindex = faiss.IndexPQ(dim2, args.pqdim, args.nbits)
            opq = faiss.OPQMatrix(dim, 1, dim2)
            print(opq.M)#debug:1
            print(opq.pq)#debug:None
            #opq.train(xs)#?#opq.is_trained
            index = faiss.IndexPreTransform(opq, subindex)
            print(opq.M)#debug:1
            print(opq.pq)#debug:None
        elif args.quantization == "none":
            index = faiss.IndexFlatL2(dim)
    elif algo=="ivf":
        if args.quantization == "pq":
            quantizer = faiss.IndexFlatL2(dim)
            index = faiss.IndexIVFPQ(quantizer, dim, args.nlists, args.pqdim, args.nbits)
        elif args.quantization == "opq":
            dim2 = dim // args.pqdim * args.pqdim
            quantizer = faiss.IndexFlatL2(dim)
            subindex = faiss.IndexIVFPQ(quantizer, dim2, args.nlists, args.pqdim, args.nbits)
            opq = faiss.OPQMatrix(dim, 1, dim2)
            index = faiss.IndexPreTransform(opq, subindex)
        elif args.quantization == "none":
            quantizer = faiss.IndexFlatL2(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, args.nlists)
    elif algo=="ivf-hnsw":
        if args.quantization == "pq":
            quantizer = faiss.IndexHNSWFlat(dim, args.degree)
            index = faiss.IndexIVFPQ(quantizer, dim, args.nlists, args.pqdim, args.nbits)
        elif args.quantization == "opq":
            dim2 = dim // args.pqdim * args.pqdim
            quantizer = faiss.IndexHNSWFlat(dim, args.degree)
            subindex = faiss.IndexIVFPQ(quantizer, dim2, args.nlists, args.pqdim, args.nbits)
            opq = faiss.OPQMatrix(dim, 1, dim2)
            index = faiss.IndexPreTransform(opq, subindex)
        elif args.quantization == "none":
            quantizer = faiss.IndexHNSWFlat(dim, args.degree)
            index = faiss.IndexIVFFlat(quantizer, dim, args.nlists)
    elif algo=='ivf-prq':
        if args.quantization == "pq":
            quantizer = faiss.IndexFlatL2(dim)
            index = faiss.IndexIVFProductResidualQuantizer(quantizer, dim, args.nlists, args.nsplits, args.pqdim, args.nbits)#TODO
        elif args.quantization == "opq":
            dim2 = dim // args.pqdim * args.pqdim
            quantizer = faiss.IndexFlatL2(dim)
            subindex = faiss.IndexIVFPQ(quantizer, dim2, args.nlists, args.pqdim, args.nbits)#TODO
            opq = faiss.OPQMatrix(dim, 1, dim2)
            index = faiss.IndexPreTransform(opq, subindex)
        elif args.quantization == "none":
            print("Error: ivf-prq must use quantization.")
            exit(0)


    if args.refineRatio>1.0:
        index = faiss.IndexRefineFlat(index)
        index.k_factor = args.refineRatio#TODO:why recall decrease?
    
    if args.gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
        if isinstance(index, faiss.GpuIndex):
            print("\nTransfer to GPU. The total number of GPUs is ", faiss.get_num_gpus())
        else:
            print("Transfering to GPU failed. The chosen index is not supported.")
    
    if not (algo == "brute" and args.quantization == "none"):
        print("\ntraining index...")
        t0 = time.time()
        index.train(xs)
        print(time.strftime("[%H:%M:%S]", time.localtime()), f"Done. Train time: {time.time() - t0 :.2f} seconds")
        
    print("\nbuilding index...")
    t0 = time.time()
    index.add(xb) # debug
    # if args.gpu:
    #     for i0, xs in dataset_iterator(xb, preproc, add_batch_size):
    #         i1 = i0 + xs.shape[0]
    #         index.add_with_ids(xs, np.arange(i0, i1))
    #         if max_add > 0 and index.ntotal > max_add:
    #             print("Flush indexes to CPU")
    #             for i in range(ngpu):
    #                 index_src_gpu = faiss.downcast_index(index.at(i))
    #                 index_src = faiss.index_gpu_to_cpu(index_src_gpu)
    #                 print("  index %d size %d" % (i, index_src.ntotal))
    #                 index_src.copy_subset_to(indexall, 0, 0, nb)
    #                 index_src_gpu.reset()
    #                 index_src_gpu.reserveMemory(max_add)
    #             index.sync_with_shard_indexes()
    # else:
    #     index.add(xb)
    print(time.strftime("[%H:%M:%S]", time.localtime()), f"Done. Build time: {time.time() - t0 :.2f} seconds")
    
    return index

def searchDetail(index, xq, K, gt):
    print(f"\nsearching {K}-nn with index {args.algo}...")
    t0 = time.time()
    _, I = index.search(xq, K)
    print(time.strftime("[%H:%M:%S]", time.localtime()), f"Done. Search time: {time.time() - t0 :.2f} seconds")
    # calculate recall
    if not args.noRecall:
        print("\nrecall:")
        calc_recall(I, gt)
    #save
    #if args.outfile is not None:


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
    print("K =", K)
    print(args) # show arguments
    xb, xq, gt, dim = load_dataset()
    
    # append dim of value 0
    if args.n0dim != 0:
        print("\nInserting 0...")
        na = args.n0dim
        subDim = (dim + na) // args.pqdim
        xb = insert_zeros(xb, subDim, na)
        xq = insert_zeros(xq, subDim, na)
        print(time.strftime("[%H:%M:%S]", time.localtime()), "new base shape:", xb.shape)
        dim += na
     
    xs = xb[np.random.choice(xb.shape[0], size=min(xb.shape[0], args.numSample), replace=False),:]

    # PCA
    if args.decomposition == "pca":
        print("\nfitting pca...")
        mat = faiss.PCAMatrix (dim, args.ncomponents)
        mat.train(xs)
        dim = args.ncomponents
        xs = mat.apply(xb)
        xb = mat.apply(xb)
        xq = mat.apply(xq)
        print(time.strftime("[%H:%M:%S]", time.localtime()), "new base shape:", xb.shape)
    
    # build
    index = build(dim, xb, xs)
    
    # search
    search(index, xq, K, gt)
