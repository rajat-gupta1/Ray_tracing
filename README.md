# Ray Tracing

Time taken by CPU for 100M: 290s
Time taken by GPU for 100M: 4s

Best configuration for running the GPU code: 128 threads/Block. However, 32, 64 and 256 also gave similar performances. 

The code is used to generate the image of an object as seen by an observer looking through his window. The code asks user to provide input for n and Number of rays respectively and the remaining values have been assumed to be constant in the code. 

The graph generated for different values of n and Nrays are shown below:

![Fig 1](./plot_100_6.png)
![Fig 2](./plot_500_6.png)
![Fig 3](./plot_1000_7.png)
![Fig 4](./plot_1000_8.png)

## Code to execute

Below is the code to execute from the terminal

./executable 1000 1000000
