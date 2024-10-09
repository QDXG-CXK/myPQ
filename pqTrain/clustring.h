/* Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <random>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>
#include "assert.h"
#include <memory>
#include <string>
#include <typeinfo>
#include <type_traits>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

static float L2Sqr(const void *pVect1v, const void *pVect2v, const void *qty_ptr)
{
    float *pVect1 = (float *)pVect1v;
    float *pVect2 = (float *)pVect2v;
    size_t qty = *((size_t *)qty_ptr);

    float res = 0;
    for (size_t i = 0; i < qty; i++) {
        float t = *pVect1 - *pVect2;
        pVect1++;
        pVect2++;
        res += t * t;
    }
    return (res);
}

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
static float L2Sqr_Neon(const void *pVect1v, const void *pVect2v, const void *qty_ptr)
{
    float *pVect1 = (float *)pVect1v;
    float *pVect2 = (float *)pVect2v;
    size_t qty = *((size_t *)qty_ptr);

    float32x4_t sum_vec = vdupq_n_f32(0);
    size_t i = 0;
    for (; i + 4 <= qty; i += 4) {
        float32x4_t a_vec = vld1q_f32(pVect1 + i);
        float32x4_t b_vec = vld1q_f32(pVect2 + i);
        float32x4_t diff_vec = vsubq_f32(a_vec, b_vec);
        sum_vec = vfmaq_f32(sum_vec, diff_vec, diff_vec);
    }
    float32_t sum = vaddvq_f32(sum_vec);
    for (; i < qty; ++i) {
        float32_t diff = pVect1[i] - pVect2[i];
        sum += diff * diff;
    }
    return sum;
}
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
static float L2SqrSIMD4ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr)
{
    size_t qty = *((size_t *)qty_ptr);
    size_t qty4 = qty >> 2 << 2;

    float res = L2Sqr_Neon(pVect1v, pVect2v, &qty4);
    size_t qty_left = qty - qty4;

    float *pVect1 = (float *)pVect1v + qty4;
    float *pVect2 = (float *)pVect2v + qty4;
    float res_tail = L2Sqr(pVect1, pVect2, &qty_left);

    return (res + res_tail);
}
#endif

template <typename MTYPE>
using DISTFUNC = MTYPE (*)(const void *, const void *, const void *);

template <typename MTYPE>
class SpaceInterface {
public:
    // virtual void search(void *);
    virtual size_t get_data_size() = 0;

    virtual DISTFUNC<MTYPE> get_dist_func() = 0;

    virtual void *get_dist_func_param() = 0;

    virtual ~SpaceInterface()
    {}
};

class L2Space : public SpaceInterface<float> {
    DISTFUNC<float> fstdistfunc_;
    size_t data_size_;
    size_t dim_;
    std::string fstdistfunc_name_;

public:
    L2Space(size_t dim)
    {
        fstdistfunc_ = L2Sqr;
        fstdistfunc_name_ = "L2Sqr";
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
        if (dim % 4 == 0) {
            fstdistfunc_ = L2Sqr_Neon;
            fstdistfunc_name_ = "L2Sqr_Neon";
        }

        if (dim % 4 != 0) {
            fstdistfunc_ = L2SqrSIMD4ExtResiduals;
            fstdistfunc_name_ = "L2SqrSIMD4ExtResiduals";
        }

#endif
        dim_ = dim;
        data_size_ = dim * sizeof(float);
    }

    size_t get_data_size()
    {
        return data_size_;
    }

    std::string get_fstdistfunc_name()
    {
        return fstdistfunc_name_;
    }

    DISTFUNC<float> get_dist_func()
    {
        return fstdistfunc_;
    }

    void *get_dist_func_param()
    {
        return &dim_;
    }

    ~L2Space()
    {}
};

struct RandomGenerator {
    std::mt19937 mt;

    int64_t seed = 1234;

    /// random positive integer
    int rand_int();

    /// generate random integer between 0 and max-1
    int rand_int(int max);

    float rand_float();

    RandomGenerator(int64_t seed = 1234);

    // explicit RandomGenerator(int64_t seed = 1234);
};

/* random permutation */
void rand_perm(int *perm, size_t n, int64_t seed);

/* Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#define EPS (1 / 1024.)

int RandomGenerator::rand_int()
{
    return mt() & 0x7fffffff;
}

int RandomGenerator::rand_int(int max)
{
    return mt() % max;
}

float RandomGenerator::rand_float()
{
    return mt() / float(mt.max());
}

RandomGenerator::RandomGenerator(int64_t seed) : seed(seed)
{}

template <class T>
struct Kmeans {
    size_t d;  ///< dimension of the vectors
    size_t k;  ///< nb of centroids
               /// number of clustering iterations
    int niter = 25;
    /// redo clustering this many times and keep the clusters with the best
    /// objective
    int nredo = 1;

    /// seed for the random number generator
    int seed = 1234;

    /// when the training set is encoded, batch size of the codec decoder
    size_t decode_block_size = 32768;

    /** centroids (k * d)
     * if centroids are set on input to train, they will be used as
     * initialization
     */
    std::vector<float> centroids;

    Kmeans(int d, int k);

    virtual std::vector<float> train(uint32_t n, const T *x);

    virtual ~Kmeans()
    {}
};

template <typename T>
Kmeans<T>::Kmeans(int d, int k) : d(d), k(k)
{}

template <typename T>
std::vector<float> Kmeans<T>::train(uint32_t vector_num, const T *vector_data)

{
    assert(vector_num >= k);

    if (typeid(T).name() == typeid(float).name()) {
        for (size_t i = 0; i < vector_num * d; i++) {
            if (!std::isfinite(vector_data[i])) {
                std::cout << "Vector data input is not finite!" << std::endl;
            }
        }
    }

    std::vector<T> raw_centroids;
    if (vector_num == k) {

        std::cout << "Warning! The number of training points is equal to number of clusters" << std::endl;
        // centroids.resize(d * k);

        raw_centroids.resize(d * k);

        memcpy(raw_centroids.data(), vector_data, sizeof(T) * d * k);
        std::vector<float> temp_centroids(raw_centroids.begin(), raw_centroids.end());
        centroids = temp_centroids;

        return centroids;
    }

    std::cout << "Clustering " << vector_num << " points in " << d << "D to " << k << " clusters, "
              << "redo " << nredo << " times, " << niter << " iterations" << std::endl;

    std::unique_ptr<uint32_t[]> assign(new uint32_t[vector_num]);
    std::unique_ptr<float[]> dis(new float[vector_num]);

    std::vector<float> best_centroids;
    std::vector<float> zero_centroids(d * k, 0);
    double best_obj = std::numeric_limits<double>::max();

    L2Space space(d);
    auto fstdistfunc_ = space.get_dist_func();
    auto dist_func_param_ = space.get_dist_func_param();

    std::cout << "Using dist function : " << space.get_fstdistfunc_name() << std::endl;

    for (int redo = 0; redo < nredo; redo++) {
        std::cout << "redo: " << redo << std::endl;
        // initialize (remaining) centroids with random points from the dataset
        // 使用随机数据点初始化质心
        raw_centroids.resize(d * k, 0);
        // centroids.resize(d * k);
        std::vector<int> perm(vector_num);

        rand_perm(perm.data(), vector_num, seed + 1 + redo * 15486557L);

        std::cout << "rand_perm ok!" << std::endl;

        for (int i = 0; i < k; i++) {
            memcpy(&raw_centroids[i * d], vector_data + perm[i] * d, sizeof(T) * d);
        }

        centroids.resize(d * k, 0);
        for (int i = 0; i < d * k; i++) {
            centroids[i] = (float)(raw_centroids[i]);
        }

        // for (int i = 0; i < k; i++) {
        //     for (int j = 0; j < d; j++) {
        //         std::cout << centroids[i * d + j] << " ";
        //     }
        //     std::cout << std::endl;
        // }
        std::cout << "memcpy centroids ok!" << std::endl;
        // k-means iterations
        double obj = 0;

        // std::vector<T> raw_vector_cache(d, 0);

        // float* vector_data_p = vector_data;

        for (int i = 0; i < niter; i++) {

            // 寻找距离每个vector最近的质心
            // std::cout << "niter: " << i << std::endl;
            for (uint32_t id = 0; id < vector_num; id++) {

                // data类型如果不是float
                if (typeid(T).name() != typeid(float).name()) {
                    // memcpy(raw_vector_cache.data(), vector_data + d * id, sizeof(T) * d);
                    std::vector<float> float_vector_cache(d, 0);
                    for (int kk = 0; kk < d; kk++) {
                        float_vector_cache[kk] = (float)(*(vector_data + d * id + kk));
                    }

                    float *vector_data_p = float_vector_cache.data();
                    std::pair<float, uint32_t> neareat_c(
                        std::numeric_limits<float>::max(), std::numeric_limits<uint32_t>::max());
                    for (uint32_t c_id = 0; c_id < k; c_id++) {
                        float dist = fstdistfunc_(vector_data_p, centroids.data() + d * c_id, dist_func_param_);
                        // std::cout << id << " " << c_id << " " << dist << std::endl;
                        if (dist < neareat_c.first) {
                            neareat_c.first = dist;
                            neareat_c.second = c_id;
                        }
                    }
                    // std::cout << neareat_c.first << " " << neareat_c.second << std::endl;
                    dis[id] = neareat_c.first;
                    assign[id] = neareat_c.second;
                } else {
                    const T *vector_data_p = vector_data + d * id;
                    std::pair<float, uint32_t> neareat_c(
                        std::numeric_limits<float>::max(), std::numeric_limits<uint32_t>::max());
                    for (uint32_t c_id = 0; c_id < k; c_id++) {
                        float dist = fstdistfunc_(vector_data_p, centroids.data() + d * c_id, dist_func_param_);
                        // std::cout << id << " " << c_id << " " << dist << std::endl;
                        if (dist < neareat_c.first) {
                            neareat_c.first = dist;
                            neareat_c.second = c_id;
                        }
                    }
                    // std::cout << neareat_c.first << " " << neareat_c.second << std::endl;
                    dis[id] = neareat_c.first;
                    assign[id] = neareat_c.second;
                }
            }

            // 计算当前聚类质量
            obj = 0;
            for (uint32_t j = 0; j < vector_num; j++) {
                obj += dis[j];
            }
            // 更新质心
            std::vector<float> hassign(k);
            centroids = zero_centroids;
            // memset(centroids.data(), 0, sizeof(float) * d * k);
            for (uint32_t id = 0; id < vector_num; id++) {
                uint32_t ci = assign[id];
                assert(ci >= 0 && ci < k);
                float *c = centroids.data() + ci * d;
                const T *xi = (vector_data + id * d);
                hassign[ci] += 1.0;
                for (size_t j = 0; j < d; j++) {
                    // float temp = xi[j];
                    c[j] = c[j] + float(xi[j]);
                }
            }
            for (uint32_t ci = 0; ci < k; ci++) {
                if (hassign[ci] == 0) {
                    continue;
                }
                float norm = 1.0 / hassign[ci];
                float *c = centroids.data() + ci * d;
                for (size_t j = 0; j < d; j++) {
                    c[j] = c[j] * norm;
                }
            }
            /* Take care of void clusters */
            size_t nsplit = 0;
            RandomGenerator rng(seed);
            for (size_t ci = 0; ci < k; ci++) {
                if (hassign[ci] == 0) { /* need to redefine a centroid */
                    size_t cj;
                    for (cj = 0; true; cj = (cj + 1) % k) {
                        /* probability to pick this cluster for split */
                        float p = (hassign[cj] - 1.0) / (float)(vector_num - k);
                        float r = rng.rand_float();
                        if (r < p) {
                            break; /* found our cluster to be split */
                        }
                    }
                    memcpy(centroids.data() + ci * d, centroids.data() + cj * d, sizeof(*centroids.data()) * d);

                    /* small symmetric pertubation */
                    for (size_t j = 0; j < d; j++) {
                        if (j % 2 == 0) {
                            centroids[ci * d + j] *= 1 + EPS;
                            centroids[cj * d + j] *= 1 - EPS;
                        } else {
                            centroids[ci * d + j] *= 1 - EPS;
                            centroids[cj * d + j] *= 1 + EPS;
                        }
                    }

                    /* assume even split of the cluster */
                    hassign[ci] = hassign[cj] / 2;
                    hassign[cj] -= hassign[ci];
                    nsplit++;
                }
            }
        }

        if (nredo > 1) {
            if ((obj < best_obj)) {
                best_centroids = centroids;
                best_obj = obj;
            }
        }
    }
    if (nredo > 1) {
        centroids = best_centroids;
    }
    return centroids;
}

void rand_perm(int *perm, size_t n, int64_t seed)
{
    for (size_t i = 0; i < n; i++)
        perm[i] = i;

    RandomGenerator rng(seed);

    for (size_t i = 0; i + 1 < n; i++) {
        int i2 = i + rng.rand_int(n - i);
        std::swap(perm[i], perm[i2]);
    }
}
