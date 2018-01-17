// this should really be in the implementation (.cpp file)
#include <pluginlib/class_list_macros.h>
#include "integrate_and_fire_nodelet.h"

// watch the capitalization carefully


namespace integration
{
    void integrate_and_fire_nodelet::onInit()
    {
        ROS_INFO("Initializing nodelet...");
        ros::NodeHandle& nh = getNodeHandle();
        ros::NodeHandle& local_nh = getPrivateNodeHandle();
        image_transport::ImageTransport it(nh);
        image_transport::Subscriber sub = it.subscribe("image", 10, &integrate_and_fire_nodelet::callback, this);
        pub.reset(new ros::Publisher(nh.advertise<sensor_msgs::Image>("/test", 1)));
    }

    void integrate_and_fire_nodelet::callback(const sensor_msgs::ImageConstPtr& msg){
        cv_bridge::CvImagePtr cv_ptr;
        try
        {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
            cv::Mat img = cv_ptr->image;

            sensor_msgs::ImagePtr img_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", img).toImageMsg();
            pub->publish(img_msg);

        }
        catch (cv_bridge::Exception& e)
        {
          ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
        }
      }
}
//PLUGINLIB_EXPORT_CLASS(vocus2::integrate_and_fire_nodelet, nodelet::Nodelet)
PLUGINLIB_DECLARE_CLASS(integrate_and_fire_nodelet, integrate_and_fire_nodelet, integration::integrate_and_fire_nodelet, nodelet::Nodelet);
