#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif


#include <nodelet/nodelet.h>
#include <iostream>
#include <sstream>
#include <unistd.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sys/stat.h>
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <boost/shared_ptr.hpp>
#include "opencv2/xfeatures2d.hpp"
//#include "integrate.cuh"
#include <cuda.h>

using namespace cv;
using namespace std;

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)


    struct neuron{
      float v_rest = -70;
      float R = 100;
      float tauv = 10;
      float x;
      float y;
      float radius;
      //V = v_rest
      float V = -70;
      neuron(float x_coord, float y_coord, float r):x(x_coord), y(y_coord), radius(r)
      {}
    };


namespace integration
{
  // dVm = 1/tauv*(-Vm+Vrest+R*Iinj);


    class integrate_and_fire_nodelet : public nodelet::Nodelet
    {
        public:
          virtual void onInit();
          void callback(const sensor_msgs::ImageConstPtr& msg);
          void build_grid(int x_size, int y_size, float radius);
          void inject_input();
        private:
          //boost::shared_ptr<ros::Publisher> pub;
          image_transport::Subscriber sub;

          //std::vector<std::vector<neuron> > ngrid;
          std::vector<cv::KeyPoint> prevKeypoints;
          cv::Mat img;
          double millisecond;

    };

}
