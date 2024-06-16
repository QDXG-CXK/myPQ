

// void PQsearch(
//         uint32_t k,
//         uint32_t nbits,
//         uint32_t nq,
//         const float* xq,
//         uint32_t nb,
//         const uint8_t *codes,
//         float* distances,
//         uint32_t* labels)
// {
//     // float_maxheap_array_t res = {size_t(n), size_t(k), labels, distances};
//     // pq.search(x, n, codes.data(), ntotal, &res, true);
//     Heap heap;
//     for (uint32_t i=0; i<nq; i++){
//         //create lut
//         std::unique_ptr<float[]> dis_tables(new float[nx * ksub * M]);
//         compute_distance_tables(nx, x, dis_tables.get());

//         //compute distances and insert into heap
//         pq_knn_search_with_tables<CMax<float, int64_t>>(
//             *this,
//             nbits,
//             dis_tables.get(),
//             codes,
//             ncodes,
//             res,
//             init_finalize_heap);

//         //save knn ids and distances
//     }
// }


#include <iostream>
#include <cassert>
#include "utils/Heap.h" // Include the myPQ heap header

void test_maxheap_operations() {
    size_t k = 5; // heap size
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

    // Fill the heap
    for (size_t i = 0; i < k; ++i) {
        heap_array.addn(1, &values[i], ids[i]);
    }

    // Reorder to get sorted output
    heap_array.reorder();

    // Check ordering (should be max heap, so in decreasing order)
    float last_val = std::numeric_limits<float>::max();
    for (size_t i = 0; i < k; ++i) {
        float current_val = heap_array.val[i];
        std::cout << "Value: " << current_val << ", ID: " << heap_array.ids[i] << std::endl;
        assert(current_val <= last_val); // validate order
        last_val = current_val;
    }
}

int main() {
    try {
        test_maxheap_operations();
        std::cout << "Test completed successfully." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
