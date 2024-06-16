#include <iostream>
#include <cassert>
#include "utils/Heap.h" // Include the myPQ heap header

void test_maxheap_operations() {
    float last_val;
    size_t k = 3; // heap size
    myPQ::float_maxheap_array_t heap_array;
    heap_array.nh = 1; // number of heaps
    heap_array.k = k; // elements per heap
    heap_array.val = new float[heap_array.nh * heap_array.k];
    heap_array.ids = new int64_t[heap_array.nh * heap_array.k];

    // Initialize heap
    heap_array.heapify();

    // Test data
    float values[] = {3.0, 1.0, 4.0, 1.5, 2.0};
    int64_t ids[] = {3, 1, 4, 2, 5};

    for (int i=0;i<5;i++){
        std::cout<<i<<":\n";
        heap_array.addn_with_ids(1, &values[i], &ids[i]);
        // Check ordering (should be max heap, so in decreasing order)
        last_val = std::numeric_limits<float>::max();
        for (size_t i = 0; i < k; ++i) {
            float current_val = heap_array.val[i];
            std::cout << "Value: " << current_val << ", ID: " << heap_array.ids[i] << std::endl;
            //assert(current_val <= last_val); // validate order
            last_val = current_val;
        }
    }

    std::cout<<"reorder:\n";
        // Reorder to get sorted output
        heap_array.reorder();
        // Check ordering (should be max heap, so in decreasing order)
        last_val = std::numeric_limits<float>::max();
        for (size_t i = 0; i < k; ++i) {
            float current_val = heap_array.val[i];
            std::cout << "Value: " << current_val << ", ID: " << heap_array.ids[i] << std::endl;
            //assert(current_val <= last_val); // validate order
            last_val = current_val;
        }
}

void PQsearch(
        uint32_t k,
        uint32_t nbits,
        uint32_t pqdim,
        uint32_t dsub,
        float* centroids,
        uint32_t nq,
        const float* xq,
        uint32_t ncodes,
        const uint8_t *codes,
        float* distances,
        uint32_t* labels)
{
    const uint32_t ksub = 1 << nbits;
    const uint32_t dim = pqdim * dsub;

    for (uint32_t i=0; i<nq; i++){
        //create lut
        float *dis_table = new float[ksub * pqdim];
        const float *query = xq + i * dim;
        for (size_t m = 0; m < pqdim; m++) {
            for(size_t k = 0; k < ksub; k++){
                const float *x = query + m * dsub;
                const float *y = centroids + m * dsub * ksub + k;
                float dis = 0;
                for(size_t d = 0; d < dsub; d++){
                    const float tmp =x[d] - y[d];
                    dis += tmp * tmp;
                }
                dis_table[m * ksub + k] = dis;
            }
        }

        
        //compute distances and insert into heap
        myPQ::float_maxheap_array_u32t heap = {1, k, labels + i * k, distances + i * k};
        heap.heapify();
        switch (nbits) {
            case 8:
                const uint8_t *cur_code = codes;
                for (uint32_t j = 0; j < ncodes; j++) {
                    float dis = 0;
                    float *cur_lut = dis_table;
                    for (int m = 0; m < pqdim; m++) {
                        dis += cur_lut[*cur_code];
                        cur_code++;
                        cur_lut += ksub;
                    }
                    heap.addn_with_ids(1, &dis, &j);
                }
                break;
            default:
                assert(0);
        }
        heap.reorder(); // heap --> ordered list
    }
}

int main() {
    //input
    const uint32_t k = 2;
    const uint32_t nbits = 8;
    const uint32_t pqdim = 2;
    const uint32_t dsub = 2;
    float centroids[pqdim * 256 * dsub]={};
    const uint32_t nq = 10;
    float xq[nq * pqdim * dsub]={};
    const uint32_t ncodes = 10;
    uint8_t codes[ncodes * pqdim] = {};

    //output
    float* distances = new float[nq * k];
    uint32_t* labels = new uint32_t[nq * k];


    //debug
    PQsearch(k, nbits, pqdim, dsub, centroids, nq, xq, ncodes, codes,distances, labels);

    return 0;
}
