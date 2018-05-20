

#include <iostream>
#include <math.h>
//#include <stdio>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "integrate_and_fire_nodelet.h"
using namespace std;
using namespace cv;


__global__
void integrate(vector<vector<neuron> > ngrid, float tauv, Mat img)
{
  int c_index = blockIdx.x;
  int r_index = blockIdx.x*blockDim.x + threadIdx.x;
  int stride = blockDim.x*gridDim.x;

  /*for (int i = c_index; i < ngrid.size(); i+= 1){
    for( j = r_index; j < ngrid[0].size; j += stride){
      neuron temp = ngrid[i][j];
    }
  }*/
}


__global__
void add(int n, float *x, float *y)
{
  int index = blockIdx.x*blockDim.x + threadIdx.x;
  int stride = blockDim.x*gridDim.x;
  for (int i = index; i < n; i+= stride)
      y[i] = x[i] + y[i];
}
