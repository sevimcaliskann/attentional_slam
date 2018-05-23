/*****************************************************************************
*
* main.cpp file for the saliency program VOCUS2.
* A detailed description of the algorithm can be found in the paper: "Traditional Saliency Reloaded: A Good Old Model in New Shape", S. Frintrop, T. Werner, G. Martin Garcia, in Proceedings of the IEEE International Conference on Computer Vision and Pattern Recognition (CVPR), 2015.
* Please cite this paper if you use our method.
*
* Implementation:	  Thomas Werner   (wernert@cs.uni-bonn.de)
* Design and supervision: Simone Frintrop (frintrop@iai.uni-bonn.de)
*
* Version 1.1
*
* This code is published under the MIT License
* (see file LICENSE.txt for details)
*
******************************************************************************/





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
#include <geometry_msgs/PointStamped.h>
#include <dynamic_reconfigure/server.h>
#include <vocus2/vocus_paramsConfig.h>
#include <boost/shared_ptr.hpp>
#include <opencv2/plot.hpp>

#include "VOCUS2.h"

using namespace std;
using namespace cv;

struct stat sb;


int WEBCAM_MODE = 0;
bool VIDEO_MODE = false;
float MSR_THRESH = 0.75; // most salient region
bool SHOW_OUTPUT = true;
string WRITE_OUT = "";
string WRITE_PATH = "";
string OUTOUT = "";
bool WRITEPATHSET = false;
bool CENTER_BIAS = false;

float SIGMA, K;
int MIN_SIZE, METHOD;
VOCUS2_Cfg cfg;// = VOCUS2_Cfg();
VOCUS2 vocus;

//ros::Publisher pose;
boost::shared_ptr<ros::Publisher> pub;



vector<string> split_string(const string &s, char delim) {
    vector<string> elems;
    stringstream ss(s);
    string item;
    while (getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}


void callback(vocus2::vocus_paramsConfig &config, uint32_t level) {
  cfg.center_sigma = config.center_sigma;
  cfg.c_space = (ColorSpace)config.c_space;
  cfg.fuse_conspicuity = (FusionOperation)config.fuse_conspicuity;
  cfg.fuse_feature = (FusionOperation)config.fuse_feature;
  cfg.center_sigma = config.center_sigma;
  cfg.surround_sigma = config.surround_sigma;
  cfg.normalize = config.normalize;
  cfg.n_scales = config.n_scales;
  cfg.pyr_struct = (PyrStructure)config.pyr_struct;
  cfg.start_layer = config.start_layer;
  cfg.stop_layer = config.stop_layer;
  cfg.orientation = config.orientation;
  cfg.combined_features = config.combined_features;

  vocus.setCfg(cfg);
}



void imageCallback(const sensor_msgs::ImageConstPtr& msg, const VOCUS2 &vocus){
  //while(ros::ok()){
    cv_bridge::CvImagePtr cv_ptr;
    //geometry_msgs::PointStamped p_;
    try
    {


        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        cv::Mat img = cv_ptr->image;

        //Mat salmap;
        //std::vector<cv::Mat> salmap_list;

        //vocus.process(img);

        //salmap = vocus.get_salmap();

        //salmap = vocus.get_salmap();
        //if(CENTER_BIAS)
          //  vocus.add_center_bias(0.5);


        /*for(int i = 0; i<salmap_list.size(); i++){
            salmap = salmap_list[i];
            vector<Point> msr = get_msr(salmap);

            Point2f center;
            float rad;

            cv::minEnclosingCircle(msr, center, rad);
            p_.header.stamp = msg->header.stamp;
            p_.header.frame_id = "/saliency_points";
            p_.point.x = center.x;
            p_.point.y = center.y;
            p_.point.z = rad;
            pose.publish(p_);

            //cv::cvtColor(salmap, salmap, cv::COLOR_GRAY2BGR);
            circle(img_rgb, center, 10, Scalar(255,0,0),CV_FILLED, 8,0);
        }*/



        sensor_msgs::ImagePtr img_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", img).toImageMsg();
        pub->publish(img_msg);

    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
    }
  //}
}

//get most salient region
vector<Point> get_msr(Mat& salmap){
	vector<Point> msr;
  while(msr.size()<1){
    double ma;
  	Point p_ma;
  	minMaxLoc(salmap, nullptr, &ma, nullptr, &p_ma);
	  msr.push_back(p_ma);
    circle(salmap, p_ma, 5, Scalar(0), -1);

  }
  /*

	int pos = 0;
	float thresh = MSR_THRESH*ma;

	Mat considered = Mat::zeros(salmap.size(), CV_8U);
	considered.at<uchar>(p_ma) = 1;

	while(pos < (int)msr.size()){
		int r = msr[pos].y;
		int c = msr[pos].x;

		for(int dr = -1; dr <= 1; dr++){
			for(int dc = -1; dc <= 1; dc++){
				if(dc == 0 && dr == 0) continue;
				if(considered.ptr<uchar>(r+dr)[c+dc] != 0) continue;
				if(r+dr < 0 || r+dr >= salmap.rows) continue;
				if(c+dc < 0 || c+dc >= salmap.cols) continue;

				if(salmap.ptr<float>(r+dr)[c+dc] >= thresh){
					msr.push_back(Point(c+dc, r+dr));
					considered.ptr<uchar>(r+dr)[c+dc] = 1;
				}
			}
		}
		pos++;
	}
*/
	return msr;
}



int main(int argc, char* argv[]) {

    /*ros::init(argc, argv, "my_tf_broadcaster");
    ros::NodeHandle nh;


    image_transport::ImageTransport it(nh);
    //image_transport::Subscriber sub = it.subscribe("image", 1, imageCallback);
    image_transport::Subscriber sub = it.subscribe("image", 10, boost::bind(&imageCallback, _1, boost::ref(vocus)));
    pub.reset(new ros::Publisher(nh.advertise<sensor_msgs::Image>("/marked_salient_image", 1)));
    //pose = nh.advertise<geometry_msgs::PointStamped>("saliency_points", 1000);

    dynamic_reconfigure::Server<vocus2::vocus_paramsConfig> server;

    dynamic_reconfigure::Server<vocus2::vocus_paramsConfig>::CallbackType f;
    f = boost::bind(&callback, _1, _2);
    server.setCallback(f);*/




    cv::Mat img = cv::imread("/home/sevim/catkin_ws/src/vocus2/images/airport2.jpg", CV_LOAD_IMAGE_COLOR);
    //resize(img, img, Size(2*img.rows, 2*img.cols));
    //Mat img(480,640, CV_8UC3, Scalar(0, 0, 0));
    //img.at<int>(240, 240) = 255;
    //resize(img, img, Size(), 0.5, 0.5);
    //imwrite("/home/sevim/catkin_ws/src/vocus2/src/results/original.png", img);
    std::cout << "The image size : " << img.rows << ", " << img.cols << std::endl;
    vocus.process(img);
    Mat salmap = vocus.get_salmap();

    //GaussianBlur(salmap, salmap, Size(5,5), 3, 3, BORDER_REPLICATE);

    cv::namedWindow("view", WINDOW_AUTOSIZE);
    cv::imshow("view", salmap);
    cv::waitKey(3000);

    //cv::Mat copy_salmap = salmap.clone();

    //std::vector<Point> msr = get_msr(copy_salmap);

    //for(int i = 0; i<msr.size(); i++)
			  	//circle(salmap, msr[i], 5, Scalar(255), 3);


    string dir = "/home/sevim/catkin_ws/src/vocus2/src/results";
    //imwrite(dir + "/salmap.png", salmap);


    vocus.write_out(dir);
    //vocus.plot_gaussian_diff("center_surround_l");
    while(ros::ok())
      ros::spinOnce();
	return EXIT_SUCCESS;
}
