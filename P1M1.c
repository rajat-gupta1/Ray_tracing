#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#define pi 3.1415

float dot_product(float A[3], float B[3])
{
    float prod = 0;
    for (int i = 0; i < 3; i ++)
        prod += A[i] * B[i];
}

int main(int argc, char *argv[])
{
    int n, Nrays;
    n = atoi(argv[1]);
    Nrays = atoi(argv[2]);
    float phi, cost, sint;

    float Wmax = 10;
    float cond_check;
    float R = 6;
    float t;
    float mag, mag2;
    float b;

    float V[3];
    float W[3];
    float C[3] = {0, 12, 0};
    float I[3];
    float N[3];
    float S[3];
    float L[3] = {4, 4, -1};

    W[1] = 10;

    int j;
    int k;

    double **G = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++)
        G[i] = (double*)malloc(n * sizeof(double));

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            G[i][j] = 0;

    for (int i = 0; i < Nrays; i++)
    {
        while(1)
        {
            phi = (float)rand()/(float)(RAND_MAX/pi);
            cost = (float)rand()/(float)(RAND_MAX/2.0) - 1.0;
            sint = sqrt(1 - pow(cost, 2));

            V[0] = sint * cos(phi);
            V[1] = sint * sin(phi);
            V[2] = cost;

            cond_check = 0;

            for (int j = 0; j < 3; j++)
            {
                W[j] = W[1] / V[1] * V[j];
                cond_check += V[j] * C[j];
            }

            cond_check = pow(cond_check, 2);
            cond_check += pow(R, 2) - dot_product(C, C);

            if (abs(W[0]) < Wmax && abs(W[2]) < Wmax && cond_check > 0)
                break;
        }

        t = dot_product(V, C) - sqrt(cond_check);
        for (int i = 0; i < 3; i++)
            I[i] = t * V[i];
        
        mag = 0;
        mag2 = 0;
        for (int i = 0; i < 3; i++)
        {
            N[i] = I[i] - C[i];
            mag += pow(N[i], 2);
            S[i] = L[i] - I[i];
            mag2 += pow(S[i], 2);
        }

    
        mag = sqrt(mag);
        mag2 = sqrt(mag2);

        for (int i = 0; i < 3; i++)
        {
            N[i] = N[i] / mag;
            S[i] = S[i] / mag;
        }

        b = 0 > dot_product(S, N) ? 0 : dot_product(S, N);

        // printf("%f\n",b);

        j = ((W[0] + Wmax) / (2 * Wmax)) * (n);
        j = n - 1 - j;
        k = ((W[2] + Wmax) / (2 * Wmax)) * (n);

        G[j][k] = G[j][k] + b;
    }

    FILE *ofp;
    
    if ((ofp = fopen("output", "w")) == NULL) {
        puts("Error: output file invalid");
        return -1;
    }

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
            fprintf(ofp, "%e,", G[i][j]);
        fprintf(ofp, "\n");
    }

    fclose(ofp);

    for (int i = 0; i < n; i++)
    {
        free(G[i]);
    }

    free(G);
}