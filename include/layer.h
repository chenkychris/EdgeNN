#ifndef LAYER_H
#define LAYER_H
#include <bits/stdc++.h>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda.h>
#include <memory>
#include <sys/time.h>
#include <vector>
#endif
float offset = 1;
#define Offset 1
#define inputDim 227 * 227 * 3
#define c1wDim 11 * 11 * 3 * 2 * 48
#define c1N 96
#define c1outDim 2 * 55 * 55 * 48
#define p1outDim 2 * 31 * 31 * 48
#define c2wDim 5 * 5 * 48 * 2 * 128
#define c2N 256
#define c2outDim 2 * 128 * 27 * 27
#define p2outDim 2 * 15 * 15 * 128
#define c3wDim 3 * 3 * 256 * 384
#define c3N 384
#define c3outDim 2 * 13 * 13 * 192
#define c4wDim 3 * 3 * 192 * 2 * 384
#define c4N 384
#define c4outDim 2 * 13 * 13 * 192
#define c5wDim 3 * 3 * 384 * 2 * 128
#define c5N 256
#define c5outDim 2 * 13 * 13 * 128
#define p3outDim 2 * 6 * 6 * 128
#define f1wDim 6 * 6 * 256 * 2 * 2048
#define f1N 2 * 2048
#define f1outDim 4096
#define f2wDim 4096 * 4096
#define f2N 2 * 2048
#define f2outDim 4096
#define f3wDim 4096 * 1000
#define f3N 1000
#define f3outDim 1000

double gettime() {
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec * 1e-6;
}

// global mem//
double mallocStart = gettime();
__device__ __managed__ float input_a[inputDim];

__device__ __managed__ float c1_weight[c1wDim];
__device__ __managed__ float c1_bias[c1N];
__device__ __managed__ float c1_a[c1outDim];
__device__ __managed__ float c1_z[c1outDim];
__device__ __managed__ float c1_o[c1outDim];

__device__ __managed__ float p1_a[p1outDim];

__device__ __managed__ float c2_weight[c2wDim];
__device__ __managed__ float c2_bias[c2N];
__device__ __managed__ float c2_a[c2outDim];
__device__ __managed__ float c2_z[c2outDim];
__device__ __managed__ float c2_o[c2outDim];

__device__ __managed__ float p2_a[p2outDim];

__device__ __managed__ float c3_weight[c3wDim];
__device__ __managed__ float c3_bias[c3N];
__device__ __managed__ float c3_a[c3outDim];
__device__ __managed__ float c3_z[c3outDim];

__device__ __managed__ float c4_weight[c4wDim];
__device__ __managed__ float c4_bias[c4N];
__device__ __managed__ float c4_a[c4outDim];
__device__ __managed__ float c4_z[c4outDim];

__device__ __managed__ float c5_weight[c5wDim];
__device__ __managed__ float c5_bias[c5N];
__device__ __managed__ float c5_a[c5outDim];
__device__ __managed__ float c5_z[c5outDim];

__device__ __managed__ float p3_a[p3outDim];

__device__ __managed__ float f1_weight[f1wDim];
__device__ __managed__ float f1_bias[f1N];
__device__ __managed__ float f1_a[f1outDim];
__device__ __managed__ float f1_z[f1outDim];

__device__ __managed__ float f2_weight[f2wDim];
__device__ __managed__ float f2_bias[f2N];
__device__ __managed__ float f2_a[f2outDim];
__device__ __managed__ float f2_z[f2outDim];

__device__ __managed__ float f3_weight[f3wDim];
__device__ __managed__ float f3_bias[f3N];
__device__ __managed__ float f3_a[f3outDim];
__device__ __managed__ float f3_z[f3outDim];
double mallocEnd = gettime();

const static float dt = 1.0E-01f;
const static float threshold = 1.0E-02f;

typedef struct pool_data {
    unsigned int x;
    unsigned int y;
} pool_data; // for pooling layer thread x and y
class ALayer {
  public:
    bool isLRN;
    long int M, N, O;

    float *output;
    float *preact;
    float *act_result; // for normalization

    float *bias;
    float *weight;

    float *L_output;

    float *d_output;
    float *d_preact;
    float *d_act_result; // for normalization
    float *d_weight;

    ALayer(long int M, long int N, long int O, char *arg);

    ~ALayer();

    void setOutput(float *data);
    void clear();
    void bp_clear();
    void Output_Layer(float *data);
};

// Utility CUDA kernel functions
__device__ float step_function(float v);
float step_function_cpu(float v);
__global__ void apply_step_function(float *input, float *output, const int N);
void apply_step_function_cpu(float *input, float *output, const int N);
__device__ float normalization(float *input, float u, int idx, const int O, const int N);
float normalization_cpu(float *input, float u, int idx, const int O, const int N);
__global__ void normalization_function(float *input, float *output, const int O, const int N);
void normalization_function_cpu(float *input, float *output, const int O, const int N);
__global__ void makeError(float *err, float *output, unsigned int Y, const int N);
__global__ void apply_grad(float *output, float *grad, const int N);

// Forward propagation kernels
__global__ void fp_preact_c1(float input[227][227][3], float preact[96][55][55], float weight[96][11][11][3], float *);
__global__ void fp_bias_c1(float preact[96][55][55], float bias[96]);
__global__ void fp_preact_p1(float input[96][55][55], float preact[96][31][31]);
//__global__ void fp_bias_p1(float preact[6][12][12], float bias[1]);
__global__ void fp_preact_c2(float input[96][31][31], float preact[128][27][27], float weight[128][96][5][5], float *);
__global__ void fp_bias_c2(float preact[128][27][27], float bias[128]);
__global__ void fp_preact_p2(float input[256][27][27], float preact[256][15][15]);
__global__ void fp_preact_c3(float input[256][15][15], float preact[384][13][13], float weight[384][256][3][3],
                             float *);
__global__ void fp_bias_c3(float preact[384][13][13], float bias[384]);
__global__ void fp_preact_c4(float input[384][13][13], float preact[384][13][13], float weight[384][384][3][3],
                             float *);
__global__ void fp_bias_c4(float preact[384][13][13], float bias[384]);
__global__ void fp_preact_c5(float input[384][13][13], float preact[256][13][13], float weight[256][384][3][3],
                             float *);
__global__ void fp_bias_c5(float preact[256][13][13], float bias[256]);
__global__ void fp_preact_p3(float input[256][13][13], float preact[256][6][6]);
__global__ void fp_preact_f1(float input[256][6][6], float preact[4096], float weight[4096][256][6][6], float *);
__global__ void fp_bias_f1(float preact[4096], float bias[4096]);
__global__ void fp_preact_f2(float input[4096], float preact[4096], float weight[4096][4096], float *);
__global__ void fp_bias_f2(float preact[4096], float bias[4096]);
__global__ void fp_preact_f3(float input[4096], float preact[1000], float weight[1000][4096], float *);
__global__ void fp_bias_f3(float preact[1000], float bias[1000]);


// FCNN

#define InDim 32
#define hDim 512
#define OutDim 32
#define train_cnt 50
#define test_cnt 100
#define cpurun 1
#define gpurun 1

using namespace std;

const static float dt = 0.5f;
const static float threshold = 1.0E-02f;

// global mem//
double mallocStart = gettime();
__device__ __managed__ float input_a[InDim];

__device__ __managed__ float h_weight[InDim * hDim];
__device__ __managed__ float h_bias[hDim];
__device__ __managed__ float h_a[hDim];
__device__ __managed__ float h_z[hDim];
__device__ __managed__ float h_dweight[InDim * hDim];
__device__ __managed__ float h_da[hDim];
__device__ __managed__ float h_dz[hDim];

__device__ __managed__ float output_weight[OutDim * hDim];
__device__ __managed__ float output_bias[OutDim];
__device__ __managed__ float output_a[OutDim];
__device__ __managed__ float output_z[OutDim];
__device__ __managed__ float output_dweight[OutDim * hDim];
__device__ __managed__ float output_da[OutDim];
__device__ __managed__ float output_dz[OutDim];
double mallocEnd = gettime();

class FLayer {
  public:
    int inDim, outDim;
    float *weight;
    float *bias;
    float *a;
    float *z;
    float *dweight;
    float *da;
    float *dz;

    FLayer(int, int, char *arg = NULL);
    ~FLayer();
    void setOutput0(float *);
    void clear();
    void bp_clear();
};

// Utility CUDA kernel functions
__device__ float step_function(float v);
__global__ void apply_step_function(float *input, float *output, const int N);
__global__ void makeError(float *, float *, float *, const int N = OutDim, float *err = NULL);
__global__ void apply_grad(float *output, float *grad, const int N);
// Forward Propagation
__global__ void fp_z_h(float *input, float *z, float weight[hDim][InDim], float *bias = NULL, float offset = 1);
__global__ void fp_bias_h(float *z, float *bias);
__global__ void fp_z_f(float *input, float *z, float weight[OutDim][hDim], float *bias = NULL, float offset = 1);
__global__ void fp_bias_f(float *z, float *bias);
// corun cpu //
void apply_step_function_cpu(float *input, float *output, const int N);
void makeError_cpu(float *dz, float *a, float *Y, const int N, float *err = NULL);
void apply_grad_cpu(float *output, float *grad, const int N);
void fp_z_h_cpu(float *input, float *z, float weight[hDim][InDim], float *bias, float offset = 0);
void fp_bias_h_cpu(float *z, float *bias);
void fp_z_f_cpu(float *input, float *z, float weight[OutDim][hDim], float *bias, float offset = 0);
void fp_bias_f_cpu(float *z, float *bias);


// LeNet
#define DEVICE 0
#define gpurun 1
#define cpurun 1
#define inputDim 28 * 28
#define c1wDim 5 * 5
#define c1N 6
#define c1outDim 24 * 24 * 6
#define s1wDim 2 * 2
#define s1N 1
#define s1outDim 12 * 12 * 6
#define c2wDim 5 * 5 * 6
#define c2N 16
#define c2outDim 8 * 8 * 16
#define s2wDim 2 * 2
#define s2N 1
#define s2outDim 4 * 4 * 16
#define c3wDim 4 * 4 * 16
#define c3N 120
#define c3outDim 1 * 1 * 120
#define f1wDim 120
#define f1N 84
#define f1outDim 84
#define f2wDim 84
#define f2N 10
#define f2outDim 10

const static float dt = 1.0E-01f;
const static float threshold = 1.0E-02f;

// global mem//
double mallocStart = gettime();
__device__ __managed__ float c3_weight[c3wDim];
__device__ __managed__ float c3_bias[c3N];
__device__ __managed__ float c3_a[c3outDim];
__device__ __managed__ float c3_z[c3outDim];
__device__ __managed__ float c3_dweight[c3wDim];
__device__ __managed__ float c3_da[c3outDim];
__device__ __managed__ float c3_dz[c3outDim];

__device__ __managed__ float input_a[inputDim];

__device__ __managed__ float c1_weight[c1wDim];
__device__ __managed__ float c1_bias[c1N];
__device__ __managed__ float c1_a[c1outDim];
__device__ __managed__ float c1_z[c1outDim];
__device__ __managed__ float c1_dweight[c1wDim];
__device__ __managed__ float c1_da[c1outDim];
__device__ __managed__ float c1_dz[c1outDim];

__device__ __managed__ float s1_weight[s1wDim];
__device__ __managed__ float s1_bias[s1N];
__device__ __managed__ float s1_a[s1outDim];
__device__ __managed__ float s1_z[s1outDim];
__device__ __managed__ float s1_dweight[s1wDim];
__device__ __managed__ float s1_da[s1outDim];
__device__ __managed__ float s1_dz[s1outDim];

__device__ __managed__ float c2_weight[c2wDim];
__device__ __managed__ float c2_bias[c2N];
__device__ __managed__ float c2_a[c2outDim];
__device__ __managed__ float c2_z[c2outDim];
__device__ __managed__ float c2_dweight[c2wDim];
__device__ __managed__ float c2_da[c2outDim];
__device__ __managed__ float c2_dz[c2outDim];

__device__ __managed__ float s2_weight[s2wDim];
__device__ __managed__ float s2_bias[s2N];
__device__ __managed__ float s2_a[s2outDim];
__device__ __managed__ float s2_z[s2outDim];
__device__ __managed__ float s2_dweight[s2wDim];
__device__ __managed__ float s2_da[s2outDim];
__device__ __managed__ float s2_dz[s2outDim];

__device__ __managed__ float f1_weight[f1wDim];
__device__ __managed__ float f1_bias[f1N];
__device__ __managed__ float f1_a[f1outDim];
__device__ __managed__ float f1_z[f1outDim];
__device__ __managed__ float f1_dweight[f1wDim];
__device__ __managed__ float f1_da[f1outDim];
__device__ __managed__ float f1_dz[f1outDim];

__device__ __managed__ float f2_weight[f2wDim];
__device__ __managed__ float f2_bias[f2N];
__device__ __managed__ float f2_a[f2outDim];
__device__ __managed__ float f2_z[f2outDim];
__device__ __managed__ float f2_dweight[f2wDim];
__device__ __managed__ float f2_da[f2outDim];
__device__ __managed__ float f2_dz[f2outDim];
double mallocEnd = gettime();

class LLayer {
  public:
    int M, N, O;

    float *output; // a
    float *preact; // z

    float *bias;
    float *weight;

    float *d_output; // da
    float *d_preact; // dz
    float *d_weight; // dw

    LLayer(int M, int N, int O, int, double &);

    ~LLayer();

    void setOutput(float *data);
    void clear();
    void bp_clear();
};

// Utility CUDA kernel functions
__device__ float step_function(float v);
__global__ void apply_step_function(float *input, float *output, const int N);
__global__ void makeError(float *err, float *output, unsigned int Y, const int N);
__global__ void apply_grad(float *output, float *grad, const int N);


// ResNet
const static float dt = 1.0E-01f;

const static float threshold = 1.0E-02f;

float offset = 1;

#define Offset 1

#define inputDim 28 * 28

#define c1wDim 5 * 5 * 6
#define c1N 6
#define c1outDim 24 * 24 * 6

#define c2wDim 2 * 2 * 6
#define c2N 6
#define c2outDim 12 * 12 * 6

#define c3wDim 2 * 2 * 6
#define c3N 6
#define c3outDim 6 * 6 * 6

#define fwDim 6 * 6 * 6 * 10
#define fN 10
#define foutDim 10

#define rwDim 4 * 4 * 1
#define rN 1
#define routDim 6 * 6 * 6

double gettime() {
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec * 1e-6;
}

// global mem//
double mallocStart = gettime();
__device__ __managed__ float input_a[inputDim];

__device__ __managed__ float c1_weight[c1wDim];
__device__ __managed__ float c1_bias[c1N];
__device__ __managed__ float c1_a[c1outDim];
__device__ __managed__ float c1_z[c1outDim];

__device__ __managed__ float c2_weight[c2wDim];
__device__ __managed__ float c2_bias[c2N];
__device__ __managed__ float c2_a[c2outDim];
__device__ __managed__ float c2_z[c2outDim];

__device__ __managed__ float c3_weight[c3wDim];
__device__ __managed__ float c3_bias[c3N];
__device__ __managed__ float c3_a[c3outDim];
__device__ __managed__ float c3_z[c3outDim];

__device__ __managed__ float f_weight[fwDim];
__device__ __managed__ float f_bias[fN];
__device__ __managed__ float f_a[foutDim];
__device__ __managed__ float f_z[foutDim];

__device__ __managed__ float r_weight[rwDim];
__device__ __managed__ float r_bias[rN];
__device__ __managed__ float r_a[routDim];
__device__ __managed__ float r_z[routDim];
double mallocEnd = gettime();
#endif

class RLayer {
  public:
    int M, N, O;
    float *output;
    float *preact;
    float *bias;
    float *weight;
    float *d_output;
    float *d_preact;
    float *d_weight;
    RLayer(int M, int N, int O, char *arg);
    ~RLayer();
    void setOutput(float *data);
    void clear();
};

// Utility CUDA kernel functions

__device__ float sigmoid(float v);
float sigmoid_cpu(float v);
__global__ void apply_sigmoid(float *input, float *output, const int N);
__global__ void makeError(float *err, float *output, unsigned int Y, const int N);
__global__ void apply_grad(float *output, float *grad, const int N);
__global__ void fp_preact_c1(float input[28][28], float preact[6][24][24], float weight[6][5][5], float *, float);
__global__ void fp_preact_c2(float input[6][24][24], float preact[6][12][12], float weight[6][2][2], float *, float);
__global__ void fp_preact_c3(float input[6][12][12], float preact[6][6][6], float weight[6][2][2], float *, float);
__global__ void fp_preact_f(float input[6][6][6], float preact[10], float weight[10][6][6][6], float *, float);
__global__ void fp_preact_r(float input[6][24][24], float preact[6][6][6], float weight[1][4][4], float, float);
__global__ void fp_add_res(float preact1[6][6][6], float preact2[6][6][6]);
////////////////cpu fp kernel////////////////////////////
void fp_preact_c1_cpu(float input[28][28], float preact[6][24][24], float weight[6][5][5], float *);
void fp_preact_c2_cpu(float input[6][24][24], float preact[6][12][12], float weight[6][2][2], float *);
void fp_preact_c3_cpu(float input[6][12][12], float preact[6][6][6], float weight[6][2][2], float *);
void fp_preact_f_cpu(float input[6][6][6], float preact[10], float weight[10][6][6][6], float *);
void fp_preact_r_cpu(float input[6][24][24], float preact[6][6][6], float weight[1][4][4], float);
