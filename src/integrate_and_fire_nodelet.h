#include <nodelet/nodelet.h>
#include <iostream>
#include <sstream>
#include <unistd.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sys/stat.h>
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <boost/shared_ptr.hpp>

namespace integration
{

    class integrate_and_fire_nodelet : public nodelet::Nodelet
    {
        public:
            virtual void onInit();
            void callback(const sensor_msgs::ImageConstPtr& msg);
          private:
            boost::shared_ptr<ros::Publisher> pub;
    };

}
