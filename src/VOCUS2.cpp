/*****************************************************************************
*
* VOCUS2.cpp file for the saliency program VOCUS2.
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

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <omp.h>
#include <algorithm>
#include <string>

#include "VOCUS2.h"
using namespace std;

VOCUS2::VOCUS2(){
	// set up default config
	this->cfg = VOCUS2_Cfg();

	this->salmap_ready = false;
	this->splitted_ready = false;
	this->processed = false;
	prepare_gaussian_kernels(cfg.center_sigma);
	prepare_gabor_kernels(cfg.center_sigma);
}

VOCUS2::VOCUS2(const VOCUS2_Cfg& cfg) {
	// use given config
	this->cfg = cfg;

	// set flags
	this->salmap_ready = false;
	this->splitted_ready = false;
	this->processed = false;
}

VOCUS2::~VOCUS2() {}

void VOCUS2::setCfg(const VOCUS2_Cfg& cfg) {
  this->cfg = cfg;
  this->salmap_ready = false;
  this->splitted_ready = false;
}

void VOCUS2::write_out(string dir){
	if(!salmap_ready) return;

	double mi, ma;

	std::cout << "Writing intermediate results to directory: " << dir <<"/"<< endl;

	for(int i = 0; i < (int)on_off_gabor45.size(); i++){
		if(on_off_gabor0[i].rows>0 && on_off_gabor0[i].cols>0){
		minMaxLoc(on_off_gabor0[i], &mi, &ma);
		imwrite(dir + "/on_off_gabor0_" + to_string(i) + ".png", (on_off_gabor0[i]-mi)/(ma-mi)*255.f);
		}

		if(off_on_gabor45[i].rows>0 && off_on_gabor45[i].cols>0){
		minMaxLoc(off_on_gabor45[i], &mi, &ma);
		imwrite(dir + "/off_on_gabor45_" + to_string(i) + ".png", (off_on_gabor45[i]-mi)/(ma-mi)*255.f);
		}

		if(on_off_gabor90[i].rows>0 && on_off_gabor90[i].cols>0){
		minMaxLoc(on_off_gabor90[i], &mi, &ma);
		imwrite(dir + "/on_off_gabor90_" + to_string(i) + ".png", (on_off_gabor90[i]-mi)/(ma-mi)*255.f);
		}

		if(off_on_gabor90[i].rows>0 && off_on_gabor90[i].cols>0){
		minMaxLoc(off_on_gabor90[i], &mi, &ma);
		imwrite(dir + "/off_on_gabor90_" + to_string(i) + ".png", (off_on_gabor90[i]-mi)/(ma-mi)*255.f);
		}

		if(on_off_gabor0[i].rows>0 && on_off_gabor0[i].cols>0){
		minMaxLoc(on_off_gabor0[i], &mi, &ma);
		imwrite(dir + "/on_off_gabor0_" + to_string(i) + ".png", (on_off_gabor0[i]-mi)/(ma-mi)*255.f);
		}

		if(off_on_gabor0[i].rows>0 && off_on_gabor0[i].cols>0){
		minMaxLoc(off_on_gabor0[i], &mi, &ma);
		imwrite(dir + "/off_on_gabor0_" + to_string(i) + ".png", (off_on_gabor0[i]-mi)/(ma-mi)*255.f);
		}

		if(on_off_gabor135[i].rows>0 && on_off_gabor135[i].cols>0){
		minMaxLoc(on_off_gabor135[i], &mi, &ma);
		imwrite(dir + "/on_off_gabor135_" + to_string(i) + ".png", (on_off_gabor135[i]-mi)/(ma-mi)*255.f);
		}

		if(off_on_gabor135[i].rows>0 && off_on_gabor135[i].cols>0){
		minMaxLoc(off_on_gabor135[i], &mi, &ma);
		imwrite(dir + "/off_on_gabor135_" + to_string(i) + ".png", (off_on_gabor135[i]-mi)/(ma-mi)*255.f);
		}

	}

	for(int i = 0; i < (int)on_off_L.size(); i++){
		minMaxLoc(on_off_L[i], &mi, &ma);
		imwrite(dir + "/on_off_L_" + to_string(i) + ".png", (on_off_L[i]-mi)/(ma-mi)*255.f);

		minMaxLoc(on_off_a[i], &mi, &ma);
		imwrite(dir + "/on_off_a_" + to_string(i) + ".png", (on_off_a[i]-mi)/(ma-mi)*255.f);

		minMaxLoc(on_off_b[i], &mi, &ma);
		imwrite(dir + "/on_off_b_" + to_string(i) + ".png", (on_off_b[i]-mi)/(ma-mi)*255.f);

		minMaxLoc(off_on_L[i], &mi, &ma);
		imwrite(dir + "/off_on_L_" + to_string(i) + ".png", (off_on_L[i]-mi)/(ma-mi)*255.f);

		minMaxLoc(off_on_a[i], &mi, &ma);
		imwrite(dir + "/off_on_a_" + to_string(i) + ".png", (off_on_a[i]-mi)/(ma-mi)*255.f);

		minMaxLoc(off_on_b[i], &mi, &ma);
		imwrite(dir + "/off_on_b_" + to_string(i) + ".png", (off_on_b[i]-mi)/(ma-mi)*255.f);


	}


        /*for(int i = 0; i < (int)pyr_surround_b.size(); i++){
                minMaxLoc(pyr_surround_b[i], &mi, &ma);
                imwrite(dir + "/pyr_surround_b_" + to_string(i) + ".png", (pyr_surround_b[i]-mi)/(ma-mi)*255.f);
        }

        for(int i = 0; i < (int)pyr_center_b.size(); i++){
                minMaxLoc(pyr_center_b[i], &mi, &ma);
                imwrite(dir + "/pyr_center_b_" + to_string(i) + ".png", (pyr_center_b[i]-mi)/(ma-mi)*255.f);
        }*/

	/*vector<Mat> tmp(6);

	tmp[0] = fuse(on_off_L, cfg.fuse_feature);
	minMaxLoc(tmp[0], &mi, &ma);
	imwrite(dir + "/feat_on_off_L.png", (tmp[0]-mi)/(ma-mi)*255.f);

	tmp[1] = fuse(on_off_a, cfg.fuse_feature);
	minMaxLoc(tmp[1], &mi, &ma);
	imwrite(dir + "/feat_on_off_a.png", (tmp[1]-mi)/(ma-mi)*255.f);

	tmp[2] = fuse(on_off_b, cfg.fuse_feature);
	minMaxLoc(tmp[2], &mi, &ma);
	imwrite(dir + "/feat_on_off_b.png", (tmp[2]-mi)/(ma-mi)*255.f);

	tmp[3] = fuse(off_on_L, cfg.fuse_feature);
	minMaxLoc(tmp[3], &mi, &ma);
	imwrite(dir + "/feat_off_on_L.png", (tmp[3]-mi)/(ma-mi)*255.f);

	tmp[4] = fuse(off_on_a, cfg.fuse_feature);
	minMaxLoc(tmp[4], &mi, &ma);
	imwrite(dir + "/feat_off_on_a.png", (tmp[4]-mi)/(ma-mi)*255.f);

	tmp[5] = fuse(off_on_b, cfg.fuse_feature);
	minMaxLoc(tmp[5], &mi, &ma);
	imwrite(dir + "/feat_off_on_b.png", (tmp[5]-mi)/(ma-mi)*255.f);

	for(int i = 0; i < 3; i++){
		vector<Mat> tmp2;
		tmp2.push_back(tmp[i]);
		tmp2.push_back(tmp[i+3]);

		string ch = "";

		switch(i){
		case 0: ch = "L"; break;
		case 1: ch = "a"; break;
		case 2: ch = "b"; break;
		}

		Mat tmp3 = fuse(tmp2, cfg.fuse_feature);

		minMaxLoc(tmp3, &mi, &ma);
		imwrite(dir + "/conspicuity_" + ch + ".png", (tmp3-mi)/(ma-mi)*255.f);
	}*/
}

void VOCUS2::process(const Mat& img){
	// clone the input image
	input = img.clone();

	// call process for desired pyramid strcture
	if(cfg.pyr_struct == NEW) pyramid_new(img);  // default
        else if(cfg.pyr_struct == CODI) pyramid_codi(img);
	else pyramid_classic(img);

	// set flag indicating that the pyramids are present
	this->processed = true;
	// compute center surround contrast
	center_surround_diff();

  /*if(cfg.orientation)*/	orientationWithCenterSurroundDiff();

}


void VOCUS2::gaborFilterImages(const Mat &base, Mat &out, int orientation, int scale){
	/*int filter_size = 11*cfg.center_sigma*pow(sqrt(2), scale);

  sigma = sqrt((cfg.surround_sigma*cfg.surround_sigma) - (cfg.center_sigma*cfg.center_sigma));
	sigma *= pow(sqrt(2), scale);
  float wavelength = pow(sqrt(2), scale) *1.5;
  Mat gaborKernel1 = cv::getGaborKernel(cv::Size(filter_size,filter_size), sigma, (float)orientation*M_PI/180, wavelength, 1, CV_PI);
  Mat gaborKernel2 = cv::getGaborKernel(cv::Size(filter_size,filter_size), sigma, (float)orientation*M_PI/180, wavelength, 1, CV_PI/2);

	float u = sum(mean(gaborKernel1))[0];
	subtract(gaborKernel1, u, gaborKernel1);
	u = sum(mean(gaborKernel2))[0];
	subtract(gaborKernel2, u, gaborKernel2);


	string dir_gabors = "/home/sevim/catkin_ws/src/vocus2/src/results/gabors";


	minMaxLoc(gaborKernel1, &mi, &ma);
	imwrite(dir_gabors + "/" + to_string(orientation) +
	"_scale_"+ to_string(scale) + "_phase_" + to_string(0) +  ".png", (gaborKernel1-mi)/(ma-mi)*255.f);

	minMaxLoc(gaborKernel2, &mi, &ma);
	imwrite(dir_gabors + "/" + to_string(orientation) +
	"_scale_"+ to_string(scale) + "_phase_" + to_string(90) +  ".png", (gaborKernel2-mi)/(ma-mi)*255.f);*/


	string dir_gabor_res = "/home/sevim/catkin_ws/src/vocus2/src/results/gabor_res";
	string dir_energy_res = "/home/sevim/catkin_ws/src/vocus2/src/results/energy_res";
	string dir_square_res = "/home/sevim/catkin_ws/src/vocus2/src/results/square_res";
	string dir_out_res = "/home/sevim/catkin_ws/src/vocus2/src/results/out_res";

	Mat res1, res2;
	double ma, mi;
	int index = 2*(int)orientation / 45;
	filter2D(base, res1, -1, gabor_filters[index][scale], Point(-1,-1), 0, BORDER_REPLICATE);
	filter2D(base, res2, -1, gabor_filters[index + 1][scale], Point(-1,-1), 0, BORDER_REPLICATE);
	minMaxLoc(res1, &mi, &ma);
	imwrite(dir_gabor_res + "/" + to_string(orientation) +
			"_scale_"+ to_string(scale) + "_phase_" + to_string(0) +  ".png", (res1-mi)/(ma-mi)*255.f);

	minMaxLoc(res2, &mi, &ma);
	imwrite(dir_gabor_res + "/" + to_string(orientation) +
		"_scale_"+ to_string(scale) + "_phase_" + to_string(90) +  ".png", (res2-mi)/(ma-mi)*255.f);

  multiply(res1, res1, res1);
	multiply(res2, res2, res2);

	minMaxLoc(res1, &mi, &ma);
	imwrite(dir_square_res + "/" + to_string(orientation) +
		"_scale_"+ to_string(scale) + "_phase_" + to_string(0) +  ".png", (res1-mi)/(ma-mi)*255.f);

	minMaxLoc(res2, &mi, &ma);
	imwrite(dir_square_res + "/" + to_string(orientation) +
		"_scale_"+ to_string(scale) + "_phase_" + to_string(90) +  ".png", (res2-mi)/(ma-mi)*255.f);

  add(res1, res2, res1);
	minMaxLoc(res1, &mi, &ma);
	imwrite(dir_energy_res + "/" + to_string(orientation) +
		"_scale_"+ to_string(scale) + ".png", (res1-mi)/(ma-mi)*255.f);



  sqrt(res1, out);
	minMaxLoc(out, &mi, &ma);
	imwrite(dir_out_res + "/" + to_string(orientation) +
		"_scale_"+ to_string(scale) +  ".png", (out-mi)/(ma-mi)*255.f);

}


void VOCUS2::pyramid_codi(const Mat& img){
	// clear previous results
	clear();

	// set flags
	salmap_ready = false;
	splitted_ready = false;

	// prepare input image (convert colorspace + split planes)
	planes = prepare_input(img);


	// create base pyramids
    vector<Mat> pyr_base_L, pyr_base_a, pyr_base_b, pyr_base_a2, pyr_base_b2;


#pragma omp parallel sections
	{
#pragma omp section
	pyr_base_L = build_multiscale_pyr(planes[0], 1.f);
#pragma omp section
	pyr_base_a = build_multiscale_pyr(planes[1], 1.f);
#pragma omp section
    pyr_base_a2 = build_multiscale_pyr(planes[2], 1.f);
#pragma omp section
    pyr_base_b = build_multiscale_pyr(planes[3], 1.f);
#pragma omp section
    pyr_base_b2 = build_multiscale_pyr(planes[4], 1.f);
	}

	// recompute sigmas that are needed to reach the desired
	// smoothing for center and surround
        float adapted_center_sigma = sqrt(pow(cfg.center_sigma,2)-1);
        float adapted_surround_sigma = sqrt(pow(cfg.surround_sigma,2)-1);

    std::cout << "adapted center sigma: " << adapted_center_sigma << std::endl;
    std::cout << "adapted surround sigma:" << adapted_surround_sigma << std::endl;

	// reserve space
	pyr_center_L.resize(pyr_base_L.size());
	pyr_center_a.resize(pyr_base_L.size());
	pyr_center_b.resize(pyr_base_L.size());

	pyr_surround_L.resize(pyr_base_L.size());
	pyr_surround_a.resize(pyr_base_L.size());
	pyr_surround_b.resize(pyr_base_L.size());

	// for every layer of the pyramid
	for(int o = 0; o < (int)pyr_base_L.size(); o++){

		// for all scales build the center and surround pyramids independently
#pragma omp parallel for

            float scaled_center_sigma = adapted_center_sigma*pow(2.0, (double)o/(double)pyr_base_L.size());
            float scaled_surround_sigma = adapted_surround_sigma*pow(2.0, (double)o/(double)pyr_base_L.size());
						//float scaled_center_sigma = adapted_center_sigma;
            //float scaled_surround_sigma = adapted_surround_sigma;

            GaussianBlur(pyr_base_L[o], pyr_center_L[o], Size(), scaled_center_sigma, scaled_center_sigma, BORDER_REPLICATE);
            GaussianBlur(pyr_base_L[o], pyr_surround_L[o], Size(), scaled_surround_sigma, scaled_surround_sigma, BORDER_REPLICATE);

            GaussianBlur(pyr_base_a[o], pyr_center_a[o], Size(), scaled_center_sigma, scaled_center_sigma, BORDER_REPLICATE);
            GaussianBlur(pyr_base_a2[o], pyr_surround_a[o], Size(), scaled_surround_sigma, scaled_surround_sigma, BORDER_REPLICATE);

            GaussianBlur(pyr_base_b[o], pyr_center_b[o], Size(), scaled_center_sigma, scaled_center_sigma, BORDER_REPLICATE);
            GaussianBlur(pyr_base_b2[o], pyr_surround_b[o], Size(), scaled_surround_sigma, scaled_surround_sigma, BORDER_REPLICATE);

						//pyr_center_L[o] = pyr_base_L[o]; pyr_surround_L[o] = pyr_base_L[o];
						//pyr_center_a[o] = pyr_base_a[o]; pyr_surround_a[o] = pyr_base_a[o];
						//pyr_center_b[o] = pyr_base_b[o]; pyr_surround_b[o] = pyr_base_b[o];
	}
}

void VOCUS2::pyramid_new(const Mat& img){
	// clear previous results
	clear();

	salmap_ready = false;
	splitted_ready = false;

	// prepare input image (convert colorspace + split channels)
	planes = prepare_input(img);

	// build center pyramid
#pragma omp parallel sections
	{
#pragma omp section
	pyr_center_L = build_multiscale_pyr(planes[0], (float)cfg.center_sigma);
#pragma omp section
	pyr_center_a = build_multiscale_pyr(planes[1], (float)cfg.center_sigma);
#pragma omp section
    pyr_center_b = build_multiscale_pyr(planes[3], (float)cfg.center_sigma);

    }

	// compute new surround sigma
	float adapted_sigma = sqrt(pow(cfg.surround_sigma,2)-pow(cfg.center_sigma,2));

	// reserve space
	pyr_surround_L.resize(pyr_center_L.size());
	pyr_surround_a.resize(pyr_center_a.size());
	pyr_surround_b.resize(pyr_center_b.size());

	// for all layers
	for(int o = 0; o < (int)pyr_center_L.size(); o++){

		// for all scales, compute surround counterpart
#pragma omp parallel for
        float scaled_sigma = adapted_sigma*pow(2.0, (double)o/(double)pyr_center_L.size());

        GaussianBlur(pyr_center_L[o], pyr_surround_L[o], Size(), scaled_sigma, scaled_sigma, BORDER_REPLICATE);
        GaussianBlur(pyr_center_a[o], pyr_surround_a[o], Size(), scaled_sigma, scaled_sigma, BORDER_REPLICATE);
        GaussianBlur(pyr_center_b[o], pyr_surround_b[o], Size(), scaled_sigma, scaled_sigma, BORDER_REPLICATE);
	}
}

void VOCUS2::pyramid_classic(const Mat& img){
	// clear previous results
	clear();

	salmap_ready = false;
	splitted_ready = false;
		// prepare input image (convert colorspace + split channels)
	planes = prepare_input(img);

	// compute center and surround pyramid directly but independent
#pragma omp parallel sections
{
#pragma omp section
	pyr_center_L = build_multiscale_pyr(planes[0], (float)cfg.center_sigma);

#pragma omp section
	pyr_center_a = build_multiscale_pyr(planes[1], (float)cfg.center_sigma);

#pragma omp section

    pyr_center_b = build_multiscale_pyr(planes[3], (float)cfg.center_sigma);


#pragma omp section
	pyr_surround_L = build_multiscale_pyr(planes[0], (float)cfg.surround_sigma);

#pragma omp section
    pyr_surround_a = build_multiscale_pyr(planes[2], (float)cfg.surround_sigma);

#pragma omp section
    pyr_surround_b = build_multiscale_pyr(planes[4], (float)cfg.surround_sigma);


}

}

void VOCUS2::prepare_gaussian_kernels(float sigma){
	int num_layer = cfg.stop_layer-cfg.start_layer+1;
	gaussian_filters.resize(num_layer);
	#pragma omp parallel for
	for(int o = 0; o < num_layer; o++){
	    float sig = pow(sqrt(2.0), o)*sigma;
	    int filter_size = 5*pow(sqrt(2), o);
			if(filter_size%2==0) filter_size += 1;
			Mat kernelX = getGaussianKernel(filter_size, sig);
			Mat kernelY = getGaussianKernel(filter_size, sig);
			Mat filt = kernelX * kernelY.t();
			gaussian_filters[o] = filt;
	}

}

void VOCUS2::prepare_gabor_kernels(float sigma){

	gabor_filters.resize(8);
	int num_layer = cfg.stop_layer-cfg.start_layer+1;
	int count = 0;
	#pragma omp parallel for
	for(int orientation = 0; orientation < 180; orientation += 45){
		gabor_filters[count].resize(num_layer);
		gabor_filters[count + 1].resize(num_layer);
		#pragma omp parallel for
		for(int i = 0; i < num_layer; i++){
			int filter_size = 22*cfg.center_sigma*pow(sqrt(2), i);
			//sigma = sqrt((cfg.surround_sigma*cfg.surround_sigma) - (cfg.center_sigma*cfg.center_sigma));
			//sigma *= pow(sqrt(2), i);
			sigma = 2.0*pow(sqrt(2), i);
		  float wavelength = pow(sqrt(2), i) *2.0;
		  Mat gaborKernel1 = cv::getGaborKernel(cv::Size(filter_size,filter_size), sigma, (float)orientation*M_PI/180, wavelength, 1, 0, CV_64F);
		  Mat gaborKernel2 = cv::getGaborKernel(cv::Size(filter_size,filter_size), sigma, (float)orientation*M_PI/180, wavelength, 1, CV_PI/2, CV_64F);

			/*double m;
			minMaxLoc(gaborKernel1, nullptr, &m);
			divide(gaborKernel1, m, gaborKernel1);
			minMaxLoc(gaborKernel2, nullptr, &m);
			divide(gaborKernel2, m, gaborKernel2);*/

			float u = sum(mean(gaborKernel1))[0];
			subtract(gaborKernel1, u, gaborKernel1);
			u = sum(mean(gaborKernel2))[0];
			subtract(gaborKernel2, u, gaborKernel2);



			gabor_filters[count][i] = gaborKernel1;
			gabor_filters[count + 1][i] = gaborKernel2;

			cout << "gabor filter phase 0 sum: " << sum(mean(gaborKernel1))[0] << endl;
			cout << "gabor filter phase 90 sum: " << sum(mean(gaborKernel2))[0] << endl;


			string dir_gabors = "/home/sevim/catkin_ws/src/vocus2/src/results/gabors";
			double ma, mi;
			minMaxLoc(gaborKernel1, &mi, &ma);
			imwrite(dir_gabors + "/" + to_string(orientation) +
			"_scale_"+ to_string(i) + "_phase_" + to_string(0) +  ".png", (gaborKernel1-mi)/(ma-mi)*255.f);

			minMaxLoc(gaborKernel2, &mi, &ma);
			imwrite(dir_gabors + "/" + to_string(orientation) +
			"_scale_"+ to_string(i) + "_phase_" + to_string(90) +  ".png", (gaborKernel2-mi)/(ma-mi)*255.f);
		}
		count += 2;
	}



}

vector<Mat> VOCUS2::build_multiscale_pyr(Mat& mat, float sigma){

    Mat tmp = mat.clone();
    cv::normalize(tmp, tmp, 0, 1, NORM_MINMAX);

    vector<Mat > pyr;
    int num_layer = cfg.stop_layer-cfg.start_layer+1;
		gaussian_filters.resize(num_layer);
    pyr.resize(num_layer);


    // compute pyramid as it is done in [Lowe2004]
#pragma omp parallel for
    for(int o = 0; o < num_layer; o++){
				//filter2D(tmp, pyr[o], -1, gaussian_filters[o], Point(-1,-1), 0, BORDER_REPLICATE );
				float sig = sigma * pow(sqrt(2), o);
				GaussianBlur(tmp, pyr[o], Size(), sig , sig, BORDER_REPLICATE);
		}

		cv::Mat diff = mat != planes[4];
		if(countNonZero(diff) ==0){
			string dir = "/home/sevim/catkin_ws/src/vocus2/src/results";
			double ma, mi;
			for(int i = 0; i < (int)pyr.size(); i++){
				minMaxLoc(pyr[i], &mi, &ma);
				imwrite(dir + "/pyrs/4_pyr_" + to_string(i) + ".png", (pyr[i]-mi)/(ma-mi)*255.f);
			}
		}
    return pyr;
}

void VOCUS2::plot_gaussian_diff(string dir){
	for(int i = 2; i < 5; i++){
		for(int j = 2; j < 4; j++ ){
			cv::Rect line1(0, gaussian_filters[i].rows/2, gaussian_filters[i].cols, 1);
			Mat data1 = gaussian_filters[i](line1);
			data1.convertTo(data1, CV_64F);

			cv::Rect line2(0, gaussian_filters[i+j].rows/2, gaussian_filters[i+j].cols, 1);
			Mat data2 = gaussian_filters[i+j](line2);
			data2.convertTo(data2, CV_64F);



			int difference = abs(gaussian_filters[gaussian_filters.size()-1].cols - data2.cols);
			copyMakeBorder( data2, data2, 0, 0, (int)difference/2, difference - (int)difference/2, BORDER_CONSTANT);
			difference = abs(gaussian_filters[gaussian_filters.size()-1].cols - data1.cols);
			copyMakeBorder( data1, data1, 0, 0, (int)difference/2, difference - (int)difference/2, BORDER_CONSTANT);


			//double max1, max2, max3;
			//Point maxLoc1, maxLoc2, maxLoc3;
			//minMaxLoc(data1, nullptr, &max1, nullptr, &maxLoc1);
			//minMaxLoc(data2, nullptr, &max2, nullptr, &maxLoc2);
			//minMaxLoc(data1 - data2, nullptr, &max3, nullptr, &maxLoc3);

			Mat plot_result1, plot_result2, plot_result3;
			Ptr<plot::Plot2d> plot1, plot2, plot3;


			plot1 = plot::Plot2d::create(data1);
			plot2 = plot::Plot2d::create(data2);
			plot3= plot::Plot2d::create(data1 - data2);
			plot1->setMaxY(0.075); plot2->setMaxY(0.075); plot3->setMaxY(0.075);
			plot1->setMinY(-0.01); plot2->setMinY(-0.01); plot3->setMinY(-0.01);
			plot1->setPlotBackgroundColor(Scalar(255, 255, 255)); plot1->setPlotLineColor(Scalar(0,0,0)); plot1->setPlotGridColor(Scalar(255, 0, 0)); plot1->setInvertOrientation(true); plot1->setShowText(true); plot1->setPlotTextColor(Scalar(0, 0, 0));
			plot2->setPlotBackgroundColor(Scalar(255, 255, 255)); plot2->setPlotLineColor(Scalar(0,0,0)); plot2->setPlotGridColor(Scalar(255, 0, 0)); plot2->setInvertOrientation(true); plot2->setShowText(true); plot2->setPlotTextColor(Scalar(0, 0, 0));
			plot3->setPlotBackgroundColor(Scalar(255, 255, 255)); plot3->setPlotLineColor(Scalar(0,0,0)); plot3->setPlotGridColor(Scalar(255, 0, 0)); plot3->setInvertOrientation(true); plot3->setShowText(true); plot3->setPlotTextColor(Scalar(0, 0, 0));

			plot1->render(plot_result1);
			plot2->render(plot_result2);
			plot3->render(plot_result3);
			imwrite(dir + "/gauss_filter_" + to_string(i) + ".png", plot_result1);
			imwrite(dir + "/gauss_filter_" + to_string(i+j) + ".png", plot_result2);
			imwrite(dir + "/difference_(" + to_string(i) + " - " + to_string(i+j) + " ).png", plot_result3);
		}
	}

}


void VOCUS2::plot_gabors(string dir){
	#pragma omp parallel for
	for(int i = 0; i < gabor_filters.size(); i++){
		#pragma omp parallel for
		/// For the last gabor filter, it gives buffer overflow
		for(int j = 0; j <  gabor_filters[i].size(); j++){
			cv::Rect line(0, gabor_filters[i][j].rows/2, gabor_filters[i][j].cols, 1);
			Mat data = gabor_filters[i][j](line);
			//data.convertTo(data, CV_64F);

			int difference = abs(gabor_filters[gabor_filters.size()-1][gabor_filters[0].size()-1].cols - data.cols);
			copyMakeBorder( data, data, 0, 0, (int)difference/2, difference - (int)difference/2, BORDER_CONSTANT);

			Mat plot_result;
			Ptr<plot::Plot2d> plot;
			plot = plot::Plot2d::create(data);
			//cout << "data: \n " << data << endl;
			plot->render(plot_result);
			int orientation = (int)(i/4)*45;
			int phase = (i%2)*90;
			imwrite(dir + "/gabor_filter_scale_" + to_string(j) +  "_orientation_ " + to_string(orientation) + "_phase_" + to_string(phase) + ".png", plot_result);
		}
	}

}

void VOCUS2::center_surround_diff(){
    int on_off_size = 6;

	on_off_L.resize(on_off_size); off_on_L.resize(on_off_size);
	on_off_a.resize(on_off_size); off_on_a.resize(on_off_size);
	on_off_b.resize(on_off_size); off_on_b.resize(on_off_size);
    int pos = 0;

	// compute DoG by subtracting layers of two pyramids
	#pragma omp parallel for
    for(int o = 2; o <= (int)4; o++){
        Mat diff;
        for(int s = 1; s<=2; s++){


            // ========== L channel ==========
            diff = (pyr_center_L[o]-pyr_surround_L[o+s+2]);
            threshold(diff, diff, 0, 1, THRESH_TOZERO);
            cv::normalize(diff, on_off_L[pos], 0, 1, NORM_MINMAX);

            diff = (pyr_surround_L[o+s+2] - pyr_center_L[o]);
            threshold(diff, diff, 0, 1, THRESH_TOZERO);
            cv::normalize(diff, off_on_L[pos], 0, 1, NORM_MINMAX);

            // ========== a channel ==========
            diff = (pyr_center_a[o]-pyr_surround_a[o+s+2]);
            threshold(diff, diff, 0, 1, THRESH_TOZERO);
            cv::normalize(diff, on_off_a[pos], 0, 1, NORM_MINMAX);
            diff = (pyr_surround_a[o+s+2] - pyr_center_a[o]);
            threshold(diff, diff, 0, 1, THRESH_TOZERO);
            cv::normalize(diff, off_on_a[pos], 0, 1, NORM_MINMAX);

            // ========== b channel ==========
            diff = (pyr_center_b[o]-pyr_surround_b[o+s+2]);
            threshold(diff, diff, 0, 1, THRESH_TOZERO);
            cv::normalize(diff, on_off_b[pos], 0, 1, NORM_MINMAX);
            diff = (pyr_surround_b[o+s+2] - pyr_center_b[o]);
            threshold(diff, diff, 0, 1, THRESH_TOZERO);
            cv::normalize(diff, off_on_b[pos], 0, 1, NORM_MINMAX);

            pos++;

        }

	}
}


void VOCUS2::orientationWithCenterSurroundDiff(){

    int on_off_size = 6;


    on_off_gabor0.resize(on_off_size); off_on_gabor0.resize(on_off_size);
    on_off_gabor45.resize(on_off_size); off_on_gabor45.resize(on_off_size);
    on_off_gabor90.resize(on_off_size); off_on_gabor90.resize(on_off_size);
    on_off_gabor135.resize(on_off_size); off_on_gabor135.resize(on_off_size);
    int pos = 0;
    // compute DoG by subtracting layers of two pyramids



		#pragma omp parallel for
		    for(int o = 2; o <=4; o++){
		        for(int s = 1; s <=2; s++){
									Mat diff, tmp1, tmp2;
									// ========== 0 channel ==========
									gaborFilterImages(planes[0], tmp1, 0, o);
									gaborFilterImages(planes[0], tmp2, 0, o+s+2);
									diff = (tmp1 - tmp2);
									threshold(diff, diff, 0, 1, THRESH_TOZERO);
									normalize(diff, on_off_gabor0[pos], 0 , 1, NORM_MINMAX);
									diff = (tmp2 - tmp1);
									threshold(diff, diff, 0, 1, THRESH_TOZERO);
									normalize(diff, off_on_gabor0[pos], 0 , 1, NORM_MINMAX);


									// ========== 45 channel ==========
									gaborFilterImages(planes[0], tmp1, 45, o);
									gaborFilterImages(planes[0],tmp2, 45, o+s+2);
									diff = (tmp1 - tmp2);
									threshold(diff, diff, 0, 1, THRESH_TOZERO);
									normalize(diff, on_off_gabor45[pos], 0 , 1, NORM_MINMAX);
									diff = (tmp2 - tmp1);
									threshold(diff, diff, 0, 1, THRESH_TOZERO);
									normalize(diff, off_on_gabor45[pos], 0 , 1, NORM_MINMAX);



									// ========== 90 channel ==========
									gaborFilterImages(planes[0], tmp1, 90, o);
									gaborFilterImages(planes[0], tmp2, 90, o+s+2);
									diff = (tmp1 - tmp2);
									threshold(diff, diff, 0, 1, THRESH_TOZERO);
									normalize(diff, on_off_gabor90[pos], 0 , 1, NORM_MINMAX);
									diff = (tmp2 - tmp1);
									threshold(diff, diff, 0, 1, THRESH_TOZERO);
									normalize(diff, off_on_gabor90[pos], 0 , 1, NORM_MINMAX);




									// ========== 135 channel ==========
									gaborFilterImages(planes[0], tmp1, 135, o);
									gaborFilterImages(planes[0], tmp2, 135, o+s+2);
									diff = (tmp1 - tmp2);
									threshold(diff, diff, 0, 1, THRESH_TOZERO);
									normalize(diff, on_off_gabor135[pos], 0 , 1, NORM_MINMAX);
									diff = (tmp2 - tmp1);
									threshold(diff, diff, 0, 1, THRESH_TOZERO);
									normalize(diff, off_on_gabor135[pos], 0 , 1, NORM_MINMAX);


									pos++;

		        }

        }



}



Mat VOCUS2::get_salmap(){

	// check if center surround contrasts are computed

	if(!processed){
		cout << "Image not yet processed. Call process(Mat)." << endl;
		return Mat();
	}

	// if saliency map is already present => return it
	if(salmap_ready) return salmap;

	// intensity feature maps
	vector<Mat> feature_intensity;
	feature_intensity.push_back(fuse(on_off_L, cfg.fuse_feature));
  feature_intensity.push_back(fuse(off_on_L, cfg.fuse_feature));



	// color feature maps
	vector<Mat> feature_color;
	feature_color.push_back(fuse(on_off_a, cfg.fuse_feature));
  feature_color.push_back(fuse(off_on_a, cfg.fuse_feature));
	feature_color.push_back(fuse(on_off_b, cfg.fuse_feature));
  feature_color.push_back(fuse(off_on_b, cfg.fuse_feature));

  vector<Mat> feature_orientation;
	if(cfg.orientation){
            feature_orientation.push_back(fuse(on_off_gabor0, cfg.fuse_feature));
            feature_orientation.push_back(fuse(on_off_gabor45, cfg.fuse_feature));
            feature_orientation.push_back(fuse(on_off_gabor90, cfg.fuse_feature));
            feature_orientation.push_back(fuse(on_off_gabor135, cfg.fuse_feature));

            feature_orientation.push_back(fuse(off_on_gabor0, cfg.fuse_feature));
            feature_orientation.push_back(fuse(off_on_gabor45, cfg.fuse_feature));
            feature_orientation.push_back(fuse(off_on_gabor90, cfg.fuse_feature));
            feature_orientation.push_back(fuse(off_on_gabor135, cfg.fuse_feature));

            /*string dir = "/home/sevim/catkin_ws/src/vocus2/src/results";
            double mi, ma;
            for(int i = 0; i < (int)feature_orientation1.size(); i++){
                    minMaxLoc(feature_orientation1[i], &mi, &ma);
                    imwrite(dir + "/feature_orientation1_" + to_string(i+1) + ".png", (feature_orientation1[i]-mi)/(ma-mi)*255.f);

            }*/

    }


		// Conspicuity maps calculations
		vector<Mat> conspicuity_maps;
    Mat conspicuity1 = fuse(feature_color, cfg.fuse_feature);
    conspicuity_maps.push_back(conspicuity1);
    if(cfg.orientation) {
        Mat conspicuity2 = fuse(feature_orientation, cfg.fuse_feature);
        conspicuity_maps.push_back(conspicuity2);
    }
    Mat conspicuity3 = fuse(feature_intensity, cfg.fuse_feature);
    conspicuity_maps.push_back(conspicuity3);



	// saliency map
	salmap = fuse(conspicuity_maps, cfg.fuse_conspicuity);
	if(cfg.normalize){
        cv::normalize(salmap, salmap, 0, 255, NORM_MINMAX, CV_8UC1);
	}

	salmap_ready = true;

	// Get uniqueness weights for each channel and each map for debugging reasons
		for(int i = 0; i < feature_orientation.size(); i++){
				float x = compute_uniqueness_weight(feature_orientation[i], 0.5, "/home/sevim/catkin_ws/src/vocus2/src/results/weights/feature_orientation_weights_" + to_string(i) + ".txt");
				stringstream stream;
				stream << fixed << setprecision(4) << x;
				std::cout << "feature_orientation " << i << " : " << stream.str() << std::endl;
		}

    /*for(int i = 0; i < feature_orientation.size(); i++){
        float x = compute_uniqueness_weight(feature_orientation[i], 0.01, "/home/sevim/catkin_ws/src/vocus2/src/results/weights/orientation_weights_" + to_string(i) + ".txt");
        stringstream stream;
        stream << fixed << setprecision(4) << x;
        std::cout << i << "_feature_orientation: " << stream.str() << std::endl;
    }*/

		/*for(int i = 0; i < feature_intensity.size(); i++){
        float x = compute_uniqueness_weight(feature_intensity[i], 0.5);
        stringstream stream;
        stream << fixed << setprecision(4) << x;
        std::cout << i << "_feature_intensity: " << stream.str() << std::endl;
    }*/
		/*for(int i = 0; i < feature_color.size(); i++){
        float x = compute_uniqueness_weight(feature_color[i], 0.5);
        stringstream stream;
        stream << fixed << setprecision(4) << x;
        std::cout << i << "_feature_color: " << stream.str() << std::endl;
    }*/




	// normalize output to [0,1]



  string dir = "/home/sevim/catkin_ws/src/vocus2/src/results";
  double mi, ma;

	for(int i = 0; i < (int)feature_intensity.size(); i++){
  	minMaxLoc(feature_intensity[i], &mi, &ma);
    imwrite(dir + "/feature_intensity_" + to_string(i) + ".png", (feature_intensity[i]-mi)/(ma-mi)*255.f);

  }

	for(int i = 0; i < (int)feature_color.size(); i++){
  	minMaxLoc(feature_color[i], &mi, &ma);
    imwrite(dir + "/feature_color_" + to_string(i) + ".png", (feature_color[i]-mi)/(ma-mi)*255.f);

  }

	for(int i = 0; i < (int)feature_orientation.size(); i++){
  	minMaxLoc(feature_orientation[i], &mi, &ma);
    imwrite(dir + "/feature_orientation_" + to_string(i) + ".png", (feature_orientation[i]-mi)/(ma-mi)*255.f);

  }

  for(int i = 0; i < (int)conspicuity_maps.size(); i++){
  	minMaxLoc(conspicuity_maps[i], &mi, &ma);
    imwrite(dir + "/conspicuity_maps_" + to_string(i) + ".png", (conspicuity_maps[i]-mi)/(ma-mi)*255.f);

  }

	return salmap;
}

Mat VOCUS2::add_center_bias(float lambda){
	if(!salmap_ready) get_salmap();

	// center
	int cr = salmap.rows/2;
	int cc = salmap.cols/2;

	// weight saliency by gaussian
	for(int r = 0; r < salmap.rows; r++){
		float* sal_row = salmap.ptr<float>(r);

		for(int c = 0; c < salmap.cols; c++){
			float d = sqrt((r-cr)*(r-cr)+(c-cc)*(c-cc));
			float fak = exp(-lambda*d*d);
			sal_row[c] *= fak;
		}
	}

	// normalize to [0,1]
	if(cfg.normalize){
		double mi, ma;
		minMaxLoc(salmap, &mi, &ma);
		salmap = (salmap-mi)/(ma-mi);
	}

	return salmap;
}

vector<Mat> VOCUS2::get_splitted_salmap(){
	if(!processed){
		cout << "Image not yet processed. Call process(Mat)." << endl;
		return vector<Mat>(1, Mat());
	}
	if(splitted_ready) return salmap_splitted;

	salmap_splitted.resize(on_off_L.size());

	for(int o = 0; o < (int)on_off_L.size(); o++){
		Mat tmp = Mat::zeros(on_off_L[o].size(), CV_32F);

		tmp += on_off_L[o];
		tmp += off_on_L[o];
		tmp += on_off_a[o];
		tmp += off_on_a[o];
		tmp += on_off_b[o];
		tmp += off_on_b[o];

		tmp /= 6.f;

		if(cfg.normalize){
			double mi, ma;
			minMaxLoc(tmp, &mi, &ma);
			tmp = (tmp-mi)/(ma-mi);
		}

		salmap_splitted[o] = tmp;
	}

	splitted_ready = true;

	return salmap_splitted;
}

void VOCUS2::census_transform(const Mat &img, Mat &out){
	unsigned int census = 0;
 	unsigned int bit = 0;
 	int m = 3;
 	int n = 3;//window size
 	int i,j,x,y;
 	int shiftCount = 0;
	Size imgSize = img.size();
	out = Mat::zeros(imgSize, CV_8U);
 	for (x = m/2; x < imgSize.height - m/2; x++)
 	{
   	for(y = n/2; y < imgSize.width - n/2; y++)
   	{
     	census = 0;
     	shiftCount = 0;
     	for (i = x - m/2; i <= x + m/2; i++)
     	{
       	for (j = y - n/2; j <= y + n/2; j++)
       	{

         	if( shiftCount != m*n/2 )//skip the center pixel
         	{
         		census <<= 1;
         		if( img.at<float>(i,j) < img.at<float>(x,y) )//compare pixel values in the neighborhood
         			bit = 1;
         		else
         			bit = 0;
         		census = census + bit;
         //cout<<census<<" ";*/

         	}
        	shiftCount ++;
       }
     }
    //cout<<endl;

    out.ptr<uchar>(x)[y] = census;
   	}
 	}
}


float VOCUS2::compute_uniqueness_weight(Mat& img, float t, string filename){

        CV_Assert(img.channels() == 1);
				Mat temp = img.clone();
				normalize(temp, temp, 0, 1, NORM_MINMAX);

				double ma;
				Point max_pt;
				minMaxLoc(temp, nullptr, &ma, nullptr, &max_pt);
				//circle(temp, max_pt, 3, Scalar(0), -1);
				if(ma==0)
					return 0;
				vector<Point> local_maxes;
				float summed = 0;


				ofstream myfile;
				if(filename!=""){

				  myfile.open(filename);
					//cout << "it has been opened: " << myfile.is_open() << endl;
				}

				unsigned int census = 0;
			 	unsigned int bit = 0;
				Mat out;
			 	int m = 3;
			 	int n = 3;//window size
			 	int i,j,x,y;
			 	int shiftCount = 0;
				Size imgSize = img.size();
				out = Mat::zeros(imgSize, CV_8U);
			 	for (x = m/2; x < imgSize.height - m/2; x++)
			 	{
			   	for(y = n/2; y < imgSize.width - n/2; y++)
			   	{
			     	census = 0;
			     	shiftCount = 0;
			     	for (i = x - m/2; i <= x + m/2; i++)
			     	{
			       	for (j = y - n/2; j <= y + n/2; j++)
			       	{

			         	if( shiftCount != m*n/2 )//skip the center pixel
			         	{
			         		census <<= 1;
			         		if( img.at<float>(i,j) < img.at<float>(x,y) )//compare pixel values in the neighborhood
			         			bit = 1;
			         		else
			         			bit = 0;
			         		census = census + bit;
			         //cout<<census<<" ";*/

			         	}
			        	shiftCount++;
			       }
			     }
			    //cout<<endl;

			    out.ptr<uchar>(x)[y] = census;
						if(census==255){
							if(myfile.is_open())
								myfile << temp.at<float>(x,y) << "\n";
							summed +=temp.at<float>(x,y);
							local_maxes.push_back(Point(x,y));

						}
			   	}
			 	}
				if(filename!=""){
					threshold(out, out, 254, 255, THRESH_TOZERO);
					imwrite(filename.substr(0, filename.length()-4) + "_census.png", out);
				}

				myfile.close();
				if(local_maxes.size() == 0)
					return 0;

				float avg = summed / local_maxes.size();
				//std::cout << "Avg " << avg << ", max " << ma << ", sum " << summed << " number of elements " << local_maxes.size() << ", the mean is: " << sum(mean(temp))[0] << std::endl;
				return ma - avg;


}




//Fuse maps using operation
Mat VOCUS2::fuse(vector<Mat> maps, FusionOperation op){

	// resulting map that is returned
	Mat fused = Mat::zeros(maps[0].size(), CV_32F);
	int n_maps = maps.size();	// no. of maps to fuse
	vector<Mat> resized;		// temp. array to hold the resized maps
	resized.resize(n_maps);		// reserve space (needed to use openmp for parallel resizing)

	// ========== ARTIMETIC MEAN ==========

	if(op == ARITHMETIC_MEAN){
		int count = 0;
#pragma omp parallel for schedule(dynamic, 1)
		for(int i = 0; i < n_maps; i++){
			if(sum(maps[i])[0] != 0){
				count++;
      	cv::add(fused, maps[i], fused, Mat(), CV_32F);
			}
		}

		fused /= (float)count;
	}

	// ========== MAX ==========

	else if(op == MAX){

#pragma omp parallel for schedule(dynamic, 1)
                /*for(int i = 0; i < n_maps; i++){
			resize(maps[i], resized[i], fused.size(), 0, 0, INTER_CUBIC);
                }*/

		for(int i = 0; i < n_maps; i++){
#pragma omp parallel for
			for(int r = 0; r < fused.rows; r++){
                                float* row_tmp = maps[i].ptr<float>(r);
				float* row_fused = fused.ptr<float>(r);
				for(int c = 0; c < fused.cols; c++){
					row_fused[c] = max(row_fused[c], row_tmp[c]);
				}
			}
		}
	}

	// ========== UNIQUENESS WEIGHTING ==========

	else if(op == UNIQUENESS_WEIGHT){
		float weight[n_maps];
		for(int i = 0; i < n_maps; i++){
			weight[i] = compute_uniqueness_weight(maps[i]);
			if(weight[i] > 0){
                                //resize(maps[i]*weight[i], resized[i], fused.size(), 0, 0, INTER_CUBIC);
                            maps[i] = weight[i]*maps[i];
			}
		}

		float sum_weights = 0;

		for(int i = 0; i < n_maps; i++){
			if(weight[i] > 0){
				sum_weights += weight[i];
                                cv::add(fused, maps[i], fused, Mat(), CV_32F);
			}
		}

		if(sum_weights > 0) fused /= sum_weights;
	}

    cv::normalize(fused, fused, 0, 255, NORM_MINMAX);

	return fused;
}

vector<Mat> VOCUS2::prepare_input(const Mat& img){

	CV_Assert(img.channels() == 3);
	vector<Mat> planes;
    planes.resize(5);

    Mat converted;
    img.convertTo(converted, CV_32FC3);

    vector<Mat> planes_bgr;
    split(converted, planes_bgr);


    Mat luminance = (planes_bgr[0] + planes_bgr[1] + planes_bgr[2])/3*255.f;
    for(int i = 0; i<planes_bgr.size(); i++){
        planes_bgr[i] /= 255.f;
        cv::divide(planes_bgr[i], luminance, planes_bgr[i]);
        //planes_bgr[i] = planes_bgr[i] - luminance;
        //threshold(planes_bgr[i], planes_bgr[i], 0, 1, THRESH_TOZERO);

    }

    if(cfg.c_space == OPPONENT_CODI){

        planes[0] = luminance;

        planes[1] = planes_bgr[2] - planes_bgr[1];
        threshold(planes[1], planes[1], 0, 1, THRESH_TOZERO);

        planes[2] = planes_bgr[1] - planes_bgr[2];
        threshold(planes[2], planes[2], 0, 1, THRESH_TOZERO);

        planes[3] = planes_bgr[0] - (planes_bgr[1] + planes_bgr[2])/2.f;
        threshold(planes[3], planes[3], 0, 1, THRESH_TOZERO);

        planes[4] = (planes_bgr[1] + planes_bgr[2])/2.f - planes_bgr[0];
        threshold(planes[4], planes[4], 0, 1, THRESH_TOZERO);


	}
	else if(cfg.c_space == OPPONENT){

				//planes_bgr[0] /= luminance;
				//planes_bgr[1] /= luminance;
				//planes_bgr[2] /= luminance;

        planes[0] = luminance;

        planes[1] = planes_bgr[2] - planes_bgr[1]+1.f;
        //threshold(planes[1], planes[1], 0, 1, THRESH_TOZERO);
        planes[1] /= 2.f;

        planes[2] = planes_bgr[1] - planes_bgr[2]+1.f;
        //threshold(planes[2], planes[2], 0, 1, THRESH_TOZERO);
        planes[2] /= 2.f;

        planes[3] = planes_bgr[0] - (planes_bgr[1] + planes_bgr[2])/2.f+1.f;
        //threshold(planes[3], planes[3], 0, 1, THRESH_TOZERO);
        planes[3] /= 2.f;

        planes[4] = (planes_bgr[1] + planes_bgr[2])/2.f - planes_bgr[0] +1.f;
        //threshold(planes[4], planes[4], 0, 1, THRESH_TOZERO);
        planes[4] /= 2.f;

	}
	else{
		Mat converted;
		img.convertTo(converted, CV_32FC3);
		converted /= 255.f;
		split(converted, planes);
	}



	return planes;
}

void VOCUS2::clear(){
	salmap.release();
	on_off_L.clear();
	off_on_L.clear();
	on_off_a.clear();
	off_on_a.clear();
	on_off_b.clear();
	off_on_b.clear();

	pyr_center_L.clear();
	pyr_surround_L.clear();
	pyr_center_a.clear();
	pyr_surround_a.clear();
	pyr_center_b.clear();
	pyr_surround_b.clear();
}
