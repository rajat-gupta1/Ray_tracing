#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#define pi 3.1415

#include <omp.h>

void write_file(int n, double **G)
{
    /* Function to write values to a file
    */
    FILE *ofp;
    
    if ((ofp = fopen("output", "w")) == NULL) {
        puts("Error: output file invalid");
    }

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
            fprintf(ofp, "%e,", G[i][j]);
        fprintf(ofp, "\n");
    }

    fclose(ofp);
}

void dot_product(double A[3], double B[3], double *dp)
{
    /* Function to calculate dot product
    */
    *dp = 0;
    for (int i = 0; i < 3; i ++)
        *dp += A[i] * B[i];
}

void scalar_product(double scl, double *vec_input, double *vec_result)
{
    /* Function to find the product between a scalar and a vector
    */

    for (int j = 0; j < 3; j++)
        vec_result[j] = scl * vec_input[j];
}

void vec_diff(double *V1, double *V2, double *vec_result)
{
    /* Function to find difference between two vectors
    */

    for (int j = 0; j < 3; j++)
        vec_result[j] = V1[j] - V2[j];
}

void unit_vec(double *V1)
{
    /* Function to convert a vector into a unit vector
    */

    // The magnitude variable
    double mag = 0;
    for (int j = 0; j < 3; j++)
        mag += pow(V1[j], 2);
    
    mag = sqrt(mag);

    for (int j = 0; j < 3; j++)
        V1[j] /= mag;
}

double LCG_random_double(u_int64_t * seed, double *rand_num)
{
    /* Function to find random number using
    Linear Congruential Generators (LCGs)
    */

    const u_int64_t m = 9223372036854775808ULL;
    const u_int64_t a = 2806196910506780709ULL;
    const u_int64_t c = 1ULL;

    *seed = (a * (*seed) + c) % m;
    *rand_num = (double) (*seed) / (double) m; 
}

u_int64_t fast_forward_LCG(u_int64_t *seed, u_int64_t n)
{
    /* Function to find random number using fast forward 
    Linear Congruential Generators (LCGs)
    */

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
}

void ray_cond (double *V, double *cond_check, double *W, double *C, double R, int loop_num, u_int64_t *seed)
{
    /* Function for the condition of path of each ray
    */

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

void matrix_gen(int Nrays, int n, double **G)
{
    /* Function for generating the matrix basis the position and the 
    intensity of rays
    */

    double Wmax = 10, cond_check, R = 6, t, b, dp;
    double V[3], W[3], C[3] = {0, 12, 0}, I[3], N[3], S[3], L[3] = {4, 4, -1};

    W[1] = 10;

    int j, k;

    // For each ray
    for (int i = 0; i < Nrays; i++)
    {
        u_int64_t seed = 213123ULL;
        while(1)
        {
            ray_cond (V, &cond_check, W, C, R, i, &seed);

            // If the ray goes out of the window
            if (abs(W[0]) < Wmax && abs(W[2]) < Wmax && cond_check > 0)
                break;
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

        // Inverting j
        j = ((W[0] + Wmax) / (2 * Wmax)) * (n);
        j = n - 1 - j;
        k = ((W[2] + Wmax) / (2 * Wmax)) * (n);

        G[j][k] = G[j][k] + b;
    }
}

void Initialise(double **G, int n)
{
    // Functino to initialise G to 0
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            G[i][j] = 0;
}

int main(int argc, char *argv[])
{
    double t1, t2;
    int n, Nrays;

    // Size of the matrix
    n = atoi(argv[1]);
    Nrays = atoi(argv[2]);

    double **G = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++)
        G[i] = (double*)malloc(n * sizeof(double));

    Initialise(G, n);

    t1 = omp_get_wtime();
    matrix_gen(Nrays, n, G);
    t2 = omp_get_wtime();

    printf("time(s): %f\n", t2 - t1);
    write_file(n, G);

    for (int i = 0; i < n; i++)
    {
        free(G[i]);
    }

    free(G);
}