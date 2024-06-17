import faiss
import numpy as np
from os.path import join

NB = 10000
NQ = 100
DIM = 32
NBITS = 8
PQDIM = 8
K = 5
OUTPUT_PATH = "./"

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
    print("NB=%d, NQ=%d, DIM=%d, NBITS=%d, PQDIM=%d, K=%d, OUTPUT_PATH=%s" % (NB, NQ, DIM, NBITS, PQDIM, K, OUTPUT_PATH))
    
    xb = np.random.rand(NB,DIM).astype(np.float32)
    xq = np.random.rand(NQ,DIM).astype(np.float32)

    # build index
    index = faiss.IndexPQ(DIM, PQDIM, NBITS)
    res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, 0, index)
    print("Training...")
    index.train(xb)
    print("Done.")
    print("Encoding...")
    index.add(xb)
    print("Done.")

    # get gt
    print("Searching...")
    D, I = index.search(xq, K)
    print("Done.")

    # save gt_D
    print("saving gt_D with shape ", D.shape)
    with open(join(OUTPUT_PATH, "gt_D.bin"), "wb") as f:
        f.write(D.tobytes())
        
    # save gt_I
    print("saving I with shape ", I.shape)
    with open(join(OUTPUT_PATH, "gt_I.bin"), "wb") as f:
        f.write(I.tobytes())
        
    # save query
    print("saving xq with shape ", xq.shape)
    with open(join(OUTPUT_PATH, "query.bin"), "wb") as f:
        f.write(xq.tobytes())

    # save codewords (M * Ksub * Dsub)
    codewords = faiss.vector_to_array(index.pq.centroids).reshape((PQDIM, 2 ** NBITS, DIM // PQDIM))
    print("saving codewords with shape ", codewords.shape)
    with open(join(OUTPUT_PATH, "mycodebook.bin"), "wb") as f:
        f.write(codewords.tobytes())
        
    # save pq codes (ntotal * code_size)
    codes = faiss.vector_to_array(index.codes).reshape((NB, PQDIM*NBITS//8))
    print("saving codes with shape ", codes.shape)
    with open(join(OUTPUT_PATH, "mycodes.bin"), "wb") as f:
        f.write(codes.tobytes())
    
