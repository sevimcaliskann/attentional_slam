// this should really be in the implementation (.cpp file)
#include <pluginlib/class_list_macros.h>
#include "integrate_and_fire_nodelet.h"

// watch the capitalization carefully


namespace integration
{
    void integrate_and_fire_nodelet::onInit()
    {
        ros::NodeHandle& nh = getNodeHandle();
        ros::NodeHandle& local_nh = getPrivateNodeHandle();
        image_transport::ImageTransport it(nh);
        sub = it.subscribe("/creative_cam/image_raw", 10, &integration::integrate_and_fire_nodelet::callback, this);
        //pub.reset(new ros::Publisher(nh.advertise<sensor_msgs::Image>("/test", 1)));
    }

    void integrate_and_fire_nodelet::callback(const sensor_msgs::ImageConstPtr& msg){
        cv_bridge::CvImagePtr cv_ptr;


        try
        {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
            img = cv_ptr->image;



            //sensor_msgs::ImagePtr img_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", img_matches).toImageMsg();

        }
        catch (cv_bridge::Exception& e)
        {
          ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
        }
      }

      void integrate_and_fire_nodelet::build_grid(int x_size, int y_size, float radius){
        /*for(int i = -radius; i<x_size + radius; i += radius){
          vector<neuron> col;
          for(int j = -radius; j < y_size + radius; j += radius){
            neuron n(i+radius, j + radius, radius);
            col.push_back(n);
          }
          ngrid.push_back(col);
        }*/
      }

      void integrate_and_fire_nodelet::inject_input(){
        Mat normalized = img.clone();
        normalize(normalized, normalized, 0, 1, NORM_MINMAX);

      }
}
//PLUGINLIB_EXPORT_CLASS(vocus2::integrate_and_fire_nodelet, nodelet::Nodelet)
PLUGINLIB_DECLARE_CLASS(integrate_and_fire_nodelet, integrate_and_fire_nodelet, integration::integrate_and_fire_nodelet, nodelet::Nodelet);
