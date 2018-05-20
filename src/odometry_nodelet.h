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

using namespace cv;

namespace vocus2
{

    class odometry_nodelet : public nodelet::Nodelet
    {
        public:
            virtual void onInit();
            void callback(const sensor_msgs::ImageConstPtr& msg);

        private:
          boost::shared_ptr<ros::Publisher> pub;
          image_transport::Subscriber sub;
          cv::Mat prevDescriptor;
          std::vector<cv::KeyPoint> prevKeypoints;
          cv::Mat prevImg;
          cv::Ptr<Feature2D> f2d;
	  

    };

}
