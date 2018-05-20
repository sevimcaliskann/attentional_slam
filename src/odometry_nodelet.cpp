// this should really be in the implementation (.cpp file)
#include <pluginlib/class_list_macros.h>
#include "odometry_nodelet.h"

// watch the capitalization carefully


namespace vocus2
{
    void odometry_nodelet::onInit()
    {




        ros::NodeHandle& nh = getNodeHandle();
        ros::NodeHandle& local_nh = getPrivateNodeHandle();
        image_transport::ImageTransport it(nh);
        sub = it.subscribe("/creative_cam/image_raw", 10, &vocus2::odometry_nodelet::callback, this);
        pub.reset(new ros::Publisher(nh.advertise<sensor_msgs::Image>("/test", 1)));
        f2d = xfeatures2d::SIFT::create();


        Mat img1 = imread("/home/sevim/Pictures/Lena1.png");
        Mat img2 = imread("/home/sevim/Pictures/lena2.png");

        std::vector<cv::KeyPoint> keypoints1;
        std::vector<cv::KeyPoint> keypoints2;

        cv::Mat descriptor1;
        cv::Mat descriptor2;

        f2d->detect(img1, keypoints1);
        f2d->compute(img1, keypoints1, descriptor1);

        f2d->detect(img2, keypoints2);
        f2d->compute(img2, keypoints2, descriptor2);

        FlannBasedMatcher matcher;
        std::vector< DMatch > matches;
        matcher.match( descriptor1, descriptor2, matches );

        double max_dist = 0; double min_dist = 100;

        //-- Quick calculation of max and min distances between keypoints
        for( int i = 0; i < descriptor1.rows; i++ )
        { double dist = matches[i].distance;
          if( dist < min_dist ) min_dist = dist;
          if( dist > max_dist ) max_dist = dist;
        }

        std::vector <DMatch> good_matches;
        for( int i = 0; i < descriptor1.rows; i++ )
        { if( matches[i].distance <= max(2*min_dist, 0.02) )
          { good_matches.push_back( matches[i]); }
        }

        Mat img_matches;
        drawMatches( img1, keypoints1, img2, keypoints2,
                     good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                     std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
        imwrite("/home/sevim/Pictures/test.png", img_matches);

    }

    void odometry_nodelet::callback(const sensor_msgs::ImageConstPtr& msg){
      ROS_INFO("inside callback");
        cv_bridge::CvImagePtr cv_ptr;
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptor;


        try
        {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
            cv::Mat img = cv_ptr->image;
            f2d->detect(img, keypoints);
            f2d->compute(img, keypoints, descriptor);


            Mat img_matches;
            if(prevKeypoints.size()>0){

                FlannBasedMatcher matcher;
                std::vector< DMatch > matches;
                matcher.match( prevDescriptor, descriptor, matches );

                double max_dist = 0; double min_dist = 100;

                //-- Quick calculation of max and min distances between keypoints
                for( int i = 0; i < prevDescriptor.rows; i++ )
                { double dist = matches[i].distance;
                  if( dist < min_dist ) min_dist = dist;
                  if( dist > max_dist ) max_dist = dist;
                }

                std::vector <DMatch> good_matches;
                for( int i = 0; i < prevDescriptor.rows; i++ )
                { if( matches[i].distance <= max(2*min_dist, 0.02) )
                  { good_matches.push_back( matches[i]); }
                }


                drawMatches( prevImg, prevKeypoints, img, keypoints,
                             good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                             std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );


          }


            sensor_msgs::ImagePtr img_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", img_matches).toImageMsg();
            pub->publish(img_msg);
            prevDescriptor = descriptor;
            prevKeypoints = keypoints;
            prevImg = img;

        }
        catch (cv_bridge::Exception& e)
        {
          ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
        }
      }
}
//PLUGINLIB_EXPORT_CLASS(vocus2::integrate_and_fire_nodelet, nodelet::Nodelet)
PLUGINLIB_DECLARE_CLASS(odometry_nodelet, odometry_nodelet, vocus2::odometry_nodelet, nodelet::Nodelet);
