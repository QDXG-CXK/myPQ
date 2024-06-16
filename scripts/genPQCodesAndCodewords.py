import faiss
import numpy as np

NBITS = 8
PQDIM = 8
N0DIM = 0
DATASET = "sift1m"
OUTPUT_PATH = "./"

if DATASET == "ood400m":
    PATH_QUERY = "/home/cuixk/dataset/newOOD/ood400m/query.i8bin"
    PATH_GROUNDTRUTH = "/home/cuixk/dataset/newOOD/ood400m/gt.u64bin"
    PATH_BASE = "/home/cuixk/dataset/newOOD/ood400m/base.i8bin"
elif DATASET == "sift1m":
    PATH_QUERY = "/home/yangshuo/data/sift1m/query.fbin"
    PATH_GROUNDTRUTH = "/home/yangshuo/data/sift1m/gt.ibin"
    PATH_BASE = "/home/yangshuo/data/sift1m/base.fbin"

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

def saveCodes(codes, fname, id_width=0):
    ntotal, code_size = codes.shape
    with open(fname, 'wb') as file:
        # Write the number of total codes as int64 at the beginning
        file.write(np.array([ntotal], dtype=np.int64).tobytes())
        if id_width > 0:
            extended_codes = np.empty((ntotal, code_size + id_width), dtype=np.uint8)
            extended_codes[:, :code_size] = codes
            for i in range(ntotal):
                # encoding id with Big-endian
                id_bytes = i.to_bytes(id_width, byteorder='little', signed=False)
                extended_codes[i, code_size:] = np.frombuffer(id_bytes, dtype=np.uint8)
            file.write(extended_codes.tobytes())
        else:
            # Write the codes array in binary format
            file.write(codes.tobytes())
    print(fname+" Saved.")

def saveCodewords(codewords, fname):
    M, Ksub, Dsub = codewords.shape
    with open(fname, 'w') as file:
        file.write(','.join(map(str, codewords.shape)) + '\n')
        for m in range(M):
            file.write(str(m) + ':\n')
            for k in range(Ksub):
                line = ','.join(map(str, codewords[m, k]))
                line += ','
                file.write(line + '\n')
    print(fname+" Saved.")

if __name__=='__main__':
    xb, xq, gt, dim = load_dataset()
    N = xb.shape[0]
    
    # append dim of value 0
    if N0DIM != 0:
        na = N0DIM
        subDim = (dim + na) // PQDIM
        xb = insert_zeros(xb, subDim, na)
        xq = insert_zeros(xq, subDim, na)
        print("new base shape:", xb.shape)
        dim += na
    
    print("Sampling...")
    x_samples = xb[np.random.choice(xb.shape[0], size=100000, replace=False),:]
    print("Done.")

    # build index
    index = faiss.IndexPQ(dim, PQDIM, NBITS)
    res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, 0, index)
    print("Training...")
    index.train(x_samples)
    print("Done.")
    print("Encoding...")
    index.add(xb)
    print("Done.")
    del xb

    get codewords (M * Ksub * Dsub)
    codewords = faiss.vector_to_array(index.pq.centroids).reshape((PQDIM, 2 ** NBITS, dim // PQDIM))
    saveCodewords(codewords, OUTPUT_PATH+"mycodewords")
    get pq codes (ntotal * code_size)
    codes = faiss.vector_to_array(index.codes).reshape((xb.shape[0], PQDIM*NBITS//8))
    saveCodes(codes, OUTPUT_PATH+"mycodes")
    