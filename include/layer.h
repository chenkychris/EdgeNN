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
#define AinputDim 227 * 227 * 3
#define Ac1wDim 11 * 11 * 3 * 2 * 48
#define Ac1N 96
#define Ac1outDim 2 * 55 * 55 * 48
#define Ap1outDim 2 * 31 * 31 * 48
#define Ac2wDim 5 * 5 * 48 * 2 * 128
#define Ac2N 256
#define Ac2outDim 2 * 128 * 27 * 27
#define Ap2outDim 2 * 15 * 15 * 128
#define Ac3wDim 3 * 3 * 256 * 384
#define Ac3N 384
#define Ac3outDim 2 * 13 * 13 * 192
#define Ac4wDim 3 * 3 * 192 * 2 * 384
#define Ac4N 384
#define Ac4outDim 2 * 13 * 13 * 192
#define Ac5wDim 3 * 3 * 384 * 2 * 128
#define Ac5N 256
#define Ac5outDim 2 * 13 * 13 * 128
#define Ap3outDim 2 * 6 * 6 * 128
#define Af1wDim 6 * 6 * 256 * 2 * 2048
#define Af1N 2 * 2048
#define Af1outDim 4096
#define Af2wDim 4096 * 4096
#define Af2N 2 * 2048
#define Af2outDim 4096
#define Af3wDim 4096 * 1000
#define Af3N 1000
#define Af3outDim 1000

double gettime() {
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec * 1e-6;
}

// global mem//
double mallocStart = gettime();
__device__ __managed__ float input_a[AinputDim];

__device__ __managed__ float c1_weight[Ac1wDim];
__device__ __managed__ float c1_bias[Ac1N];
__device__ __managed__ float c1_a[Ac1outDim];
__device__ __managed__ float c1_z[Ac1outDim];
__device__ __managed__ float c1_o[Ac1outDim];

__device__ __managed__ float p1_a[Ap1outDim];

__device__ __managed__ float c2_weight[Ac2wDim];
__device__ __managed__ float c2_bias[Ac2N];
__device__ __managed__ float c2_a[Ac2outDim];
__device__ __managed__ float c2_z[Ac2outDim];
__device__ __managed__ float c2_o[Ac2outDim];

__device__ __managed__ float p2_a[Ap2outDim];

__device__ __managed__ float c3_weight[Ac3wDim];
__device__ __managed__ float c3_bias[Ac3N];
__device__ __managed__ float c3_a[Ac3outDim];
__device__ __managed__ float c3_z[Ac3outDim];

__device__ __managed__ float c4_weight[Ac4wDim];
__device__ __managed__ float c4_bias[Ac4N];
__device__ __managed__ float c4_a[Ac4outDim];
__device__ __managed__ float c4_z[Ac4outDim];

__device__ __managed__ float c5_weight[Ac5wDim];
__device__ __managed__ float c5_bias[Ac5N];
__device__ __managed__ float c5_a[Ac5outDim];
__device__ __managed__ float c5_z[Ac5outDim];

__device__ __managed__ float p3_a[Ap3outDim];

__device__ __managed__ float f1_weight[Af1wDim];
__device__ __managed__ float f1_bias[Af1N];
__device__ __managed__ float f1_a[Af1outDim];
__device__ __managed__ float f1_z[Af1outDim];

__device__ __managed__ float f2_weight[Af2wDim];
__device__ __managed__ float f2_bias[Af2N];
__device__ __managed__ float f2_a[Af2outDim];
__device__ __managed__ float f2_z[Af2outDim];

__device__ __managed__ float f3_weight[Af3wDim];
__device__ __managed__ float f3_bias[Af3N];
__device__ __managed__ float f3_a[Af3outDim];
__device__ __managed__ float f3_z[Af3outDim];
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
#define LinputDim 28 * 28
#define Lc1wDim 5 * 5
#define Lc1N 6
#define Lc1outDim 24 * 24 * 6
#define Ls1wDim 2 * 2
#define Ls1N 1
#define Ls1outDim 12 * 12 * 6
#define Lc2wDim 5 * 5 * 6
#define Lc2N 16
#define Lc2outDim 8 * 8 * 16
#define Ls2wDim 2 * 2
#define Ls2N 1
#define Ls2outDim 4 * 4 * 16
#define Lc3wDim 4 * 4 * 16
#define Lc3N 120
#define Lc3outDim 1 * 1 * 120
#define Lf1wDim 120
#define Lf1N 84
#define Lf1outDim 84
#define Lf2wDim 84
#define Lf2N 10
#define Lf2outDim 10

const static float dt = 1.0E-01f;
const static float threshold = 1.0E-02f;

// global mem//
double mallocStart = gettime();
__device__ __managed__ float c3_weight[Lc3wDim];
__device__ __managed__ float c3_bias[Lc3N];
__device__ __managed__ float c3_a[Lc3outDim];
__device__ __managed__ float c3_z[Lc3outDim];
__device__ __managed__ float c3_dweight[Lc3wDim];
__device__ __managed__ float c3_da[Lc3outDim];
__device__ __managed__ float c3_dz[Lc3outDim];

__device__ __managed__ float input_a[LinputDim];

__device__ __managed__ float c1_weight[Lc1wDim];
__device__ __managed__ float c1_bias[Lc1N];
__device__ __managed__ float c1_a[Lc1outDim];
__device__ __managed__ float c1_z[Lc1outDim];
__device__ __managed__ float c1_dweight[Lc1wDim];
__device__ __managed__ float c1_da[Lc1outDim];
__device__ __managed__ float c1_dz[Lc1outDim];

__device__ __managed__ float s1_weight[Ls1wDim];
__device__ __managed__ float s1_bias[Ls1N];
__device__ __managed__ float s1_a[Ls1outDim];
__device__ __managed__ float s1_z[Ls1outDim];
__device__ __managed__ float s1_dweight[Ls1wDim];
__device__ __managed__ float s1_da[Ls1outDim];
__device__ __managed__ float s1_dz[Ls1outDim];

__device__ __managed__ float c2_weight[Lc2wDim];
__device__ __managed__ float c2_bias[Lc2N];
__device__ __managed__ float c2_a[Lc2outDim];
__device__ __managed__ float c2_z[Lc2outDim];
__device__ __managed__ float c2_dweight[Lc2wDim];
__device__ __managed__ float c2_da[Lc2outDim];
__device__ __managed__ float c2_dz[Lc2outDim];

__device__ __managed__ float s2_weight[Ls2wDim];
__device__ __managed__ float s2_bias[Ls2N];
__device__ __managed__ float s2_a[Ls2outDim];
__device__ __managed__ float s2_z[Ls2outDim];
__device__ __managed__ float s2_dweight[Ls2wDim];
__device__ __managed__ float s2_da[Ls2outDim];
__device__ __managed__ float s2_dz[Ls2outDim];

__device__ __managed__ float f1_weight[Lf1wDim];
__device__ __managed__ float f1_bias[Lf1N];
__device__ __managed__ float f1_a[Lf1outDim];
__device__ __managed__ float f1_z[Lf1outDim];
__device__ __managed__ float f1_dweight[Lf1wDim];
__device__ __managed__ float f1_da[Lf1outDim];
__device__ __managed__ float f1_dz[Lf1outDim];

__device__ __managed__ float f2_weight[Lf2wDim];
__device__ __managed__ float f2_bias[Lf2N];
__device__ __managed__ float f2_a[Lf2outDim];
__device__ __managed__ float f2_z[Lf2outDim];
__device__ __managed__ float f2_dweight[Lf2wDim];
__device__ __managed__ float f2_da[Lf2outDim];
__device__ __managed__ float f2_dz[Lf2outDim];
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

#define RinputDim 28 * 28

#define Rc1wDim 5 * 5 * 6
#define Rc1N 6
#define Rc1outDim 24 * 24 * 6

#define Rc2wDim 2 * 2 * 6
#define Rc2N 6
#define Rc2outDim 12 * 12 * 6

#define Rc3wDim 2 * 2 * 6
#define Rc3N 6
#define Rc3outDim 6 * 6 * 6

#define RfwDim 6 * 6 * 6 * 10
#define RfN 10
#define RfoutDim 10

#define RrwDim 4 * 4 * 1
#define RrN 1
#define RroutDim 6 * 6 * 6

double gettime() {
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec * 1e-6;
}

// global mem//
double mallocStart = gettime();
__device__ __managed__ float input_a[RinputDim];

__device__ __managed__ float c1_weight[Rc1wDim];
__device__ __managed__ float c1_bias[Rc1N];
__device__ __managed__ float c1_a[Rc1outDim];
__device__ __managed__ float c1_z[Rc1outDim];

__device__ __managed__ float c2_weight[Rc2wDim];
__device__ __managed__ float c2_bias[Rc2N];
__device__ __managed__ float c2_a[Rc2outDim];
__device__ __managed__ float c2_z[Rc2outDim];

__device__ __managed__ float c3_weight[Rc3wDim];
__device__ __managed__ float c3_bias[Rc3N];
__device__ __managed__ float c3_a[Rc3outDim];
__device__ __managed__ float c3_z[Rc3outDim];

__device__ __managed__ float f_weight[RfwDim];
__device__ __managed__ float f_bias[RfN];
__device__ __managed__ float f_a[RfoutDim];
__device__ __managed__ float f_z[RfoutDim];

__device__ __managed__ float r_weight[RrwDim];
__device__ __managed__ float r_bias[RrN];
__device__ __managed__ float r_a[RroutDim];
__device__ __managed__ float r_z[RroutDim];
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
