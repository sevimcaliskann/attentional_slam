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

ros::Publisher pose;
image_transport::Publisher pub;

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
  vocus.setCfg(cfg);
}

void print_usage(){

	cout << "===== SALIENCY =====" << endl << endl;

	cout << "   -x <name>" << "\t\t" << "Config file (is loaded first, additional options have higher priority)" << endl << endl;

	cout << "   -C <value>" << "\t\t" << "Used colorspace [default: 1]:" << endl;
	cout << "\t\t   " << "0: LAB" << endl;
	cout << "\t\t   " << "1: Opponent (CoDi)" << endl;
	cout << "\t\t   " << "2: Opponent (Equal domains)\n" << endl;

	cout << "   -f <value>" << "\t\t" << "Fusing operation (Feature maps) [default: 0]:" << endl;
	cout << "   -F <value>" << "\t\t" << "Fusing operation (Conspicuity/Saliency maps) [default: 0]:" << endl;

	cout << "\t\t   " << "0: Arithmetic mean" << endl;
	cout << "\t\t   " << "1: Max" << endl;
	cout << "\t\t   " << "2: Uniqueness weight\n" << endl;

	cout << "   -p <value>" << "\t\t" << "Pyramidal structure [default: 2]:" << endl;

	cout << "\t\t   " << "0: Two independant pyramids (Classic)" << endl;
	cout << "\t\t   " << "1: Two pyramids derived from a base pyramid (CoDi-like)" << endl;
	cout << "\t\t   " << "2: Surround pyramid derived from center pyramid (New)\n" << endl;

	cout << "   -l <value>" << "\t\t" << "Start layer (included) [default: 0]" << endl << endl;

	cout << "   -L <value>" << "\t\t" << "Stop layer (included) [default: 4]" << endl << endl;

	cout << "   -S <value>" << "\t\t" << "No. of scales [default: 2]" << endl << endl;

	cout << "   -c <value>" << "\t\t" << "Center sigma [default: 2]" << endl << endl;

	cout << "   -s <value>" << "\t\t" << "Surround sigma [default: 10]" << endl << endl;
	
	//cout << "   -r" << "\t\t" << "Use orientation [default: off]  " << endl << endl;

	cout << "   -e" << "\t\t" << "Use Combined Feature [default: off]" << endl << endl;


	cout << "===== MISC (NOT INCLUDED IN A CONFIG FILE) =====" << endl << endl;

	cout << "   -v <id>" << "\t\t" << "Webcam source" << endl << endl;

	cout << "   -V" << "\t\t" << "Video files" << endl << endl;

	cout << "   -t <value>" << "\t\t" << "MSR threshold (percentage of fixation) [default: 0.75]" << endl << endl;

	cout << "   -N" << "\t\t" << "No visualization" << endl << endl;
	
	cout << "   -o <path>" << "\t\t" << "WRITE results to specified path [default: <input_path>/saliency/*]" << endl << endl;

	cout << "   -w <path>" << "\t\t" << "WRITE all intermediate maps to an existing folder" << endl << endl;

	cout << "   -b" << "\t\t" << "Add center bias to the saliency map\n" << endl << endl;

}

bool process_arguments(VOCUS2_Cfg& cfg){

	if(MSR_THRESH < 0 || MSR_THRESH > 1){
		cerr << "MSR threshold must be in the range [0,1]" << endl;
		return false;
	}

    if(cfg.start_layer > cfg.stop_layer){
        cerr << "Start layer cannot be larger than stop layer" << endl;
        return false;
    }

	if(cfg.surround_sigma <= cfg.center_sigma){
		cerr << "Surround sigma must be positive and largen than center sigma" << endl;
		return false;
	}

	if(WEBCAM_MODE >= 0 && VIDEO_MODE){
		return false;
	}

	return true;
}


//get most salient region
vector<Point> get_msr(Mat& salmap){
	double ma;
	Point p_ma;
	minMaxLoc(salmap, nullptr, &ma, nullptr, &p_ma);

	vector<Point> msr;
	msr.push_back(p_ma);

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

	return msr;
}




void imageCallback(const sensor_msgs::ImageConstPtr& msg, VOCUS2 &vocus){
    //cv_bridge::CvImagePtr cv_ptr;
    //geometry_msgs::PointStamped p_;
    //try
    //{


        //cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
        //cv::Mat img = cv_ptr->image;

        /*Mat salmap, img_rgb;
        std::vector<cv::Mat> salmap_list;

        vocus.process(img);

        if(!CENTER_BIAS) salmap = vocus.get_salmap();
        else salmap = vocus.add_center_bias(0.0005);
        //salmap = vocus.get_salmap();
        //if(CENTER_BIAS)
          //  vocus.add_center_bias(0.5);

        salmap_list = vocus.get_splitted_salmap();

        Mat salmap2;
        salmap.convertTo(salmap2,CV_8UC1, 255);
        img_rgb = cv::Mat(salmap2.size(), CV_8UC3);
        cv::cvtColor(salmap2, img_rgb, CV_GRAY2RGB);

        for(int i = 0; i<salmap_list.size(); i++){
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
        }

            //salmap.convertTo(salmap,CV_8UC1, 255);
            //img_rgb = cv::Mat(salmap.size(), CV_8UC3);
            //cv::cvtColor(salmap, img_rgb, CV_GRAY2RGB);

            //salmap.convertTo(salmap,CV_8UC1, 255);
            //img_rgb = cv::Mat(salmap.size(), CV_8UC3);
            //cv::cvtColor(salmap, img_rgb, CV_GRAY2RGB);

            //circle(img_rgb, Point(center.x, center.y),10, Scalar(255,0,0),CV_FILLED, 8,0);
        //}


        sensor_msgs::ImagePtr img_msg = cv_bridge::CvImage(std_msgs::Header(), "rgb8", img_rgb).toImageMsg();
        pub.publish(img_msg);*/

    //}
    //catch (cv_bridge::Exception& e)
    //{
      //ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
    //}
}




int main(int argc, char* argv[]) {

    ros::init(argc, argv, "my_tf_broadcaster");

    ros::NodeHandle nh;

    /*dynamic_reconfigure::Server<vocus2::vocus_paramsConfig> server;
    dynamic_reconfigure::Server<vocus2::vocus_paramsConfig>::CallbackType f;

    f = boost::bind(&callback, _1, _2);
    server.setCallback(f);

    bool correct = process_arguments(cfg);

    if(!correct){
        print_usage();
        return EXIT_FAILURE;
    }
    //ROS_INFO("STARTED!");

    image_transport::ImageTransport it(nh);
    //image_transport::Subscriber sub = it.subscribe("image", 1, &imageCallback);
    image_transport::Subscriber sub = it.subscribe("image", 1, boost::bind(imageCallback, _1, boost::ref(vocus)));
    pub = it.advertise("/marked_salient_image", 1);
    pose = nh.advertise<geometry_msgs::PointStamped>("saliency_points", 1000);*/


    cv::Mat img = cv::imread("/home/sevim/catkin_ws/src/vocus2/images/dots.png", CV_LOAD_IMAGE_COLOR);


    vocus.process(img);
    Mat tmp = vocus.get_salmap();
    cv::normalize(tmp, tmp, 0, 255, NORM_MINMAX);
    string dir = "/home/sevim/catkin_ws/src/vocus2/src/results";
    imwrite(dir + "/salmap.png", tmp);
    //vocus.write_out(dir);
    cv::namedWindow("view", WINDOW_AUTOSIZE);
    cv::imshow("view", tmp);
    cv::waitKey(3000);

    ros::spinOnce();
	return EXIT_SUCCESS;
}


