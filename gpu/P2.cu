#include <stdio.h>
#include<cuda.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include<time.h>
#define pi 3.1415

#define MAX_BLOCKS_PER_DIM 65535
#define MIN(a,b) (((a)<(b))?(a):(b))

void write_file(int n, double *G_h)
{
    FILE *ofp;
    
    if ((ofp = fopen("output", "w")) == NULL) {
        puts("Error: output file invalid");
        // return -1;
    }

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
            fprintf(ofp, "%e,", G_h[i * n + j]);
        fprintf(ofp, "\n");
    }
    
    fclose(ofp);
}

__device__ void vec_diff(double *V1, double *V2, double *vec_result)
{
    for (int j = 0; j < 3; j++)
        vec_result[j] = V1[j] - V2[j];
}

__device__ void unit_vec(double *V1)
{
    double mag = 0;
    for (int j = 0; j < 3; j++)
        mag += pow(V1[j], 2);
    
    mag = sqrt(mag);

    for (int j = 0; j < 3; j++)
        V1[j] /= mag;
}


__device__ void dot_product(double A[3], double B[3], double *dp)
{
    *dp = 0;
    for (int i = 0; i < 3; i ++)
        *dp += A[i] * B[i];
}

__device__ void scalar_product(double scl, double *vec_input, double *vec_result)
{
    for (int j = 0; j < 3; j++)
        vec_result[j] = scl * vec_input[j];
}

__device__ double LCG_random_double(u_int64_t * seed, double *rand_num)
{
    const u_int64_t m = 9223372036854775808ULL;
    const u_int64_t a = 2806196910506780709ULL;
    const u_int64_t c = 1ULL;

    *seed = (a * (*seed) + c) % m;
    *rand_num = (double) (*seed) / (double) m; 
}

__device__ u_int64_t fast_forward_LCG(u_int64_t *seed, u_int64_t n)
{
    const u_int64_t m = 9223372036854775808ULL; 
    u_int64_t a = 2806196910506780709ULL;
    u_int64_t c = 1ULL;
    
    n = n % m;
    u_int64_t a_new = 1;
    u_int64_t c_new = 0;

    while(n > 0)
    {
        if (n & 1)
        {
            a_new *= a;
            c_new = c_new * a + c;
        }
        c *= (a + 1);
        a *= a;
        n >>= 1;
    }

    *seed = (a_new * (*seed) + c_new) % m;
    // return (a_new * seed + c_new) % m;
}

__device__ void ray_cond (double *V, double *cond_check, double *W, double *C, double R, int loop_num, u_int64_t *seed)
{
    double phi, cost, sint, dp, rand_num;

    fast_forward_LCG(seed, loop_num * 200);
    LCG_random_double(seed, &rand_num);
    phi = rand_num * pi;
    LCG_random_double(seed, &rand_num);
    cost = rand_num * 2 - 1.0;

    sint = sqrt(1 - pow(cost, 2));

    V[0] = sint * cos(phi);
    V[1] = sint * sin(phi);
    V[2] = cost;

    scalar_product(W[1] / V[1], V, W);
    dot_product(V, C, cond_check);
    *cond_check = pow(*cond_check, 2);
    dot_product (C, C, &dp);
    *cond_check += pow(R, 2) - dp;
}


__global__ void matrix_gen(int Nrays, int n, double *G)
{
    double Wmax = 10, cond_check, R = 6, t, b, dp;
    double V[3], W[3], C[3] = {0, 12, 0}, I[3], N[3], S[3], L[3] = {4, 4, -1};

    W[1] = 10;

    int j, k;
    int tid0 = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = tid0; i < Nrays; i += blockDim.x * gridDim.x) 
    {
        u_int64_t seed = 213ULL;
        while(1)
        {
            ray_cond(V, &cond_check, W, C, R, i, &seed);
            if (abs(W[0]) < Wmax && abs(W[2]) < Wmax && cond_check > 0)
                break;
            // break;
        }
        dot_product (V, C, &dp);
        t = dp - sqrt(cond_check);
        scalar_product(t, V, I);

        vec_diff(I, C, N);
        vec_diff(L, I, S);

        unit_vec(N);
        unit_vec(S);

        dot_product(S, N, &dp);
        b = 0 > dp ? 0 : dp;

        j = ((W[0] + Wmax) / (2 * Wmax)) * (n);
        j = n - 1 - j;
        k = ((W[2] + Wmax) / (2 * Wmax)) * (n);

        // printf("%f\n", b);

        atomicAdd((double *) &G[j * n + k], b);
        G[j * n + k] = G[j * n + k] + b;
    }

    // for (int i = 0; i < n * n; i++)
    //     printf("%f\n", G[i]);
}

void Initialise(double *G, int n)
{
    for (int i = 0; i < n * n; i++)
        G[i] = 0;   
}

int main(int argc, char **argv)
{

    cudaEvent_t                /* CUDA timers */
    start_device,
    stop_device;  
    float time_device;
    cudaEventCreate(&start_device);
    cudaEventCreate(&stop_device);

    int n, Nrays;
    n = atoi(argv[1]);
    Nrays = atoi(argv[2]);

    int nblocks, nthreads_per_block, nt;
    nthreads_per_block = atoi(argv[3]);
    // nt = atoi(argv[5]);

    nblocks = min(Nrays/nthreads_per_block + 1, MAX_BLOCKS_PER_DIM);

    double *G;
    cudaMalloc((void **) &G, (n * n)*sizeof(double));

    double *G_h;
    G_h = (double *) malloc(sizeof(double) * (n*n));
    
    Initialise(G_h, n);
    cudaMemcpy(G,G_h,(n*n)*sizeof(double),cudaMemcpyDeviceToHost);

    cudaEventRecord( start_device, 0 );  
    
    matrix_gen<<<nblocks, nthreads_per_block>>>(Nrays, n, G);

    cudaEventRecord( stop_device, 0 );
    cudaEventSynchronize( stop_device );
    cudaEventElapsedTime( &time_device, start_device, stop_device );

    printf("time elapsed device: %f(s)\n",  time_device/1000.);

    cudaMemcpy(G_h,G,(n*n)*sizeof(double),cudaMemcpyDeviceToHost);

    write_file(n, G_h);

    cudaFree(G_h);
    cudaFree(G);
}