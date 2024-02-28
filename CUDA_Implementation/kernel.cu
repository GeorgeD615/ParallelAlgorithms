#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>


__global__ void hello_world() {
    printf("Hello World\n");
}
int main()
{
    hello_world <<<1,1>>>();
    return 0;
}