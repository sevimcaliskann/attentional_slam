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

#include "VOCUS2.h"

using namespace cv;

VOCUS2::VOCUS2(){
	// set up default config
	this->cfg = VOCUS2_Cfg();

	// set flags indicating status of intermediate steps
	this->salmap_ready = false;
	this->splitted_ready = false;
	this->processed = false;
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

	for(int i = 0; i < (int)pyr_center_L.size(); i++){
		for(int j = 0; j < (int)pyr_center_L[i].size(); j++){
			minMaxLoc(pyr_center_L[i][j], &mi, &ma);
			imwrite(dir + "/pyr_center_L_" + to_string(i) + "_" + to_string(j) + ".png", (pyr_center_L[i][j]-mi)/(ma-mi)*255.f);

			minMaxLoc(pyr_center_a[i][j], &mi, &ma);
			imwrite(dir + "/pyr_center_a_" + to_string(i) + "_" + to_string(j) + ".png", (pyr_center_a[i][j]-mi)/(ma-mi)*255.f);

			minMaxLoc(pyr_center_b[i][j], &mi, &ma);
			imwrite(dir + "/pyr_center_b_" + to_string(i) + "_" + to_string(j) + ".png", (pyr_center_b[i][j]-mi)/(ma-mi)*255.f);

			minMaxLoc(pyr_surround_L[i][j], &mi, &ma);
			imwrite(dir + "/pyr_surround_L_" + to_string(i) + "_" + to_string(j) + ".png", (pyr_surround_L[i][j]-mi)/(ma-mi)*255.f);

			minMaxLoc(pyr_surround_a[i][j], &mi, &ma);
			imwrite(dir + "/pyr_surround_a_" + to_string(i) + "_" + to_string(j) + ".png", (pyr_surround_a[i][j]-mi)/(ma-mi)*255.f);

			minMaxLoc(pyr_surround_b[i][j], &mi, &ma);
			imwrite(dir + "/pyr_surround_b_" + to_string(i) + "_" + to_string(j) + ".png", (pyr_surround_b[i][j]-mi)/(ma-mi)*255.f);
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

	vector<Mat> tmp(6);

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
	}

	minMaxLoc(salmap, &mi, &ma);
	imwrite(dir + "/salmap.png", (salmap-mi)/(ma-mi)*255.f);
}



void VOCUS2::write_gabors(string dir){
	if(!salmap_ready) return;

	double mi, ma;

	std::cout << "Writing intermediate results to directory: " << dir <<"/"<< endl;

	for(int i = 0; i < (int)gabor.size(); i++){
		for(int j = 0; j < (int)gabor[i].size(); j++){
			minMaxLoc(gabor[i][j], &mi, &ma);
			imwrite(dir + "/gabor" + to_string(i) + "_" + to_string(j) + ".png", (gabor[i][j]-mi)/(ma-mi)*255.f);
			//imwrite(dir + "/gabor" + to_string(i) + "_" + to_string(j) + ".png", gabor[i][j]*255.f);
		}
	}
}


void VOCUS2::write_out_without_normalization(string dir){
	if(!salmap_ready) return;

	double mi, ma;

	std::cout << "Writing intermediate results to directory: " << dir <<"/"<< endl;

	for(int i = 0; i < (int)pyr_center_L.size(); i++){
		for(int j = 0; j < (int)pyr_center_L[i].size(); j++){
			imwrite(dir + "/pyr_center_L_" + to_string(i) + "_" + to_string(j) + ".png", pyr_center_L[i][j]*255.f);

			imwrite(dir + "/pyr_center_a_" + to_string(i) + "_" + to_string(j) + ".png", pyr_center_a[i][j]*255.f);

			imwrite(dir + "/pyr_center_b_" + to_string(i) + "_" + to_string(j) + ".png", pyr_center_b[i][j]*255.f);

			imwrite(dir + "/pyr_surround_L_" + to_string(i) + "_" + to_string(j) + ".png", pyr_surround_L[i][j]*255.f);

			imwrite(dir + "/pyr_surround_a_" + to_string(i) + "_" + to_string(j) + ".png", pyr_surround_a[i][j]*255.f);

			imwrite(dir + "/pyr_surround_b_" + to_string(i) + "_" + to_string(j) + ".png", pyr_surround_b[i][j]*255.f);
		}
	}

	for(int i = 0; i < (int)on_off_L.size(); i++){
		imwrite(dir + "/on_off_L_" + to_string(i) + ".png", on_off_L[i]*255.f);

		imwrite(dir + "/on_off_a_" + to_string(i) + ".png", on_off_a[i]*255.f);

		imwrite(dir + "/on_off_b_" + to_string(i) + ".png", on_off_b[i]*255.f);

		imwrite(dir + "/off_on_L_" + to_string(i) + ".png", off_on_L[i]*255.f);

		imwrite(dir + "/off_on_a_" + to_string(i) + ".png", off_on_a[i]*255.f);

		imwrite(dir + "/off_on_b_" + to_string(i) + ".png", off_on_b[i]*255.f);


	}

	vector<Mat> tmp(6);

	tmp[0] = fuse(on_off_L, cfg.fuse_feature);
	imwrite(dir + "/feat_on_off_L.png", tmp[0]*255.f);

	tmp[1] = fuse(on_off_a, cfg.fuse_feature);
	imwrite(dir + "/feat_on_off_a.png", tmp[1]*255.f);

	tmp[2] = fuse(on_off_b, cfg.fuse_feature);
	imwrite(dir + "/feat_on_off_b.png", tmp[2]*255.f);

	tmp[3] = fuse(off_on_L, cfg.fuse_feature);
	imwrite(dir + "/feat_off_on_L.png", tmp[3]*255.f);

	tmp[4] = fuse(off_on_a, cfg.fuse_feature);
	imwrite(dir + "/feat_off_on_a.png", tmp[4]*255.f);

	tmp[5] = fuse(off_on_b, cfg.fuse_feature);
	imwrite(dir + "/feat_off_on_b.png", tmp[5]*255.f);

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

		imwrite(dir + "/conspicuity_" + ch + ".png", tmp3*255.f);
	}

	imwrite(dir + "/salmap.png", salmap*255.f);
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

	if(cfg.orientation)	orientation();
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
	vector<vector<Mat> > pyr_base_L, pyr_base_a, pyr_base_b;
#pragma omp parallel sections
	{
#pragma omp section
	pyr_base_L = build_multiscale_pyr(planes[0], 1.f);
#pragma omp section
	pyr_base_a = build_multiscale_pyr(planes[1], 1.f);
#pragma omp section
	pyr_base_b = build_multiscale_pyr(planes[2], 1.f);
	}

	// recompute sigmas that are needed to reach the desired
	// smoothing for center and surround
	float adapted_center_sigma = sqrt(pow(cfg.center_sigma,2)-1);
	float adapted_surround_sigma = sqrt(pow(cfg.surround_sigma,2)-1);

	// reserve space
	pyr_center_L.resize(pyr_base_L.size());
	pyr_center_a.resize(pyr_base_L.size());
	pyr_center_b.resize(pyr_base_L.size());
	pyr_surround_L.resize(pyr_base_L.size());
	pyr_surround_a.resize(pyr_base_L.size());
	pyr_surround_b.resize(pyr_base_L.size());

	// for every layer of the pyramid
	for(int o = 0; o < (int)pyr_base_L.size(); o++){
		pyr_center_L[o].resize(cfg.n_scales);
		pyr_center_a[o].resize(cfg.n_scales);
		pyr_center_b[o].resize(cfg.n_scales);
		pyr_surround_L[o].resize(cfg.n_scales);
		pyr_surround_a[o].resize(cfg.n_scales);
		pyr_surround_b[o].resize(cfg.n_scales);

		// for all scales build the center and surround pyramids independently
#pragma omp parallel for
		for(int s = 0; s < cfg.n_scales; s++){

			float scaled_center_sigma = adapted_center_sigma*pow(2.0, (double)s/(double)cfg.n_scales);
			float scaled_surround_sigma = adapted_surround_sigma*pow(2.0, (double)s/(double)cfg.n_scales);

			GaussianBlur(pyr_base_L[o][s], pyr_center_L[o][s], Size(), scaled_center_sigma, scaled_center_sigma, BORDER_REPLICATE);
			GaussianBlur(pyr_base_L[o][s], pyr_surround_L[o][s], Size(), scaled_surround_sigma, scaled_surround_sigma, BORDER_REPLICATE);

			GaussianBlur(pyr_base_a[o][s], pyr_center_a[o][s], Size(), scaled_center_sigma, scaled_center_sigma, BORDER_REPLICATE);
			GaussianBlur(pyr_base_a[o][s], pyr_surround_a[o][s], Size(), scaled_surround_sigma, scaled_surround_sigma, BORDER_REPLICATE);

			GaussianBlur(pyr_base_b[o][s], pyr_center_b[o][s], Size(), scaled_center_sigma, scaled_center_sigma, BORDER_REPLICATE);
			GaussianBlur(pyr_base_b[o][s], pyr_surround_b[o][s], Size(), scaled_surround_sigma, scaled_surround_sigma, BORDER_REPLICATE);
		}
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
	pyr_center_b = build_multiscale_pyr(planes[2], (float)cfg.center_sigma);
	}

	// compute new surround sigma
	float adapted_sigma = sqrt(pow(cfg.surround_sigma,2)-pow(cfg.center_sigma,2));

	// reserve space
	pyr_surround_L.resize(pyr_center_L.size());
	pyr_surround_a.resize(pyr_center_a.size());
	pyr_surround_b.resize(pyr_center_b.size());

	// for all layers
	for(int o = 0; o < (int)pyr_center_L.size(); o++){
		pyr_surround_L[o].resize(cfg.n_scales);
		pyr_surround_a[o].resize(cfg.n_scales);
		pyr_surround_b[o].resize(cfg.n_scales);

		// for all scales, compute surround counterpart
#pragma omp parallel for
		for(int s = 0; s < cfg.n_scales; s++){
			float scaled_sigma = adapted_sigma*pow(2.0, (double)s/(double)cfg.n_scales);

			GaussianBlur(pyr_center_L[o][s], pyr_surround_L[o][s], Size(), scaled_sigma, scaled_sigma, BORDER_REPLICATE);
			GaussianBlur(pyr_center_a[o][s], pyr_surround_a[o][s], Size(), scaled_sigma, scaled_sigma, BORDER_REPLICATE);
			GaussianBlur(pyr_center_b[o][s], pyr_surround_b[o][s], Size(), scaled_sigma, scaled_sigma, BORDER_REPLICATE);
		}
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
	pyr_center_b = build_multiscale_pyr(planes[2], (float)cfg.center_sigma);

#pragma omp section
	pyr_surround_L = build_multiscale_pyr(planes[0], (float)cfg.surround_sigma);

#pragma omp section
	pyr_surround_a = build_multiscale_pyr(planes[1], (float)cfg.surround_sigma);

#pragma omp section
	pyr_surround_b = build_multiscale_pyr(planes[2], (float)cfg.surround_sigma);
}
}

void VOCUS2::center_surround_diff(){
	//int on_off_size = pyr_center_L.size()*cfg.n_scales;
	int on_off_size = 6;

	on_off_L.resize(on_off_size); off_on_L.resize(on_off_size);
	on_off_a.resize(on_off_size); off_on_a.resize(on_off_size);
	on_off_b.resize(on_off_size); off_on_b.resize(on_off_size);

	// compute DoG by subtracting layers of two pyramids
	int pos = 0;
	for(int o = 2; o <=4; o++){
#pragma omp parallel for
		for(int s = 2; s <=3; s++){
			Mat diff;
			//int pos = o*cfg.n_scales+s;
			// ========== L channel ==========
			Mat tmp;
			resize(pyr_surround_L[o+s][0], pyr_surround_L[o+s][0], pyr_center_L[o][0].size());
			diff = pyr_center_L[o][0]-pyr_surround_L[o+s][0];
			threshold(diff, on_off_L[pos], 0, 1, THRESH_TOZERO);
			diff *= -1.f;
			threshold(diff, off_on_L[pos], 0, 1, THRESH_TOZERO);

			resize(pyr_surround_a[o+s][0], pyr_surround_a[o+s][0], pyr_center_a[o][0].size());
			// ========== a channel ==========
			diff = pyr_center_a[o][0]-pyr_surround_a[o+s][0];
			threshold(diff, on_off_a[pos], 0, 1, THRESH_TOZERO);
			diff *= -1.f;
			threshold(diff, off_on_a[pos], 0, 1, THRESH_TOZERO);

			resize(pyr_surround_b[o+s][0], pyr_surround_b[o+s][0], pyr_center_b[o][0].size());
			// ========== b channel ==========
			diff = pyr_center_b[o][0]-pyr_surround_b[o+s][0];
			threshold(diff, on_off_b[pos], 0, 1, THRESH_TOZERO);
			diff *= -1.f;
			threshold(diff, off_on_b[pos], 0, 1, THRESH_TOZERO);
			pos++;
		}
	}
}


/*void VOCUS2::center_surround_diff(){
	int on_off_size = pyr_center_L.size()*cfg.n_scales;

	on_off_L.resize(on_off_size); off_on_L.resize(on_off_size);
	on_off_a.resize(on_off_size); off_on_a.resize(on_off_size);
	on_off_b.resize(on_off_size); off_on_b.resize(on_off_size);

	// compute DoG by subtracting layers of two pyramids
	for(int o = 0; o < (int)pyr_center_L.size(); o++){
#pragma omp parallel for
		for(int s = 0; s < cfg.n_scales; s++){
			Mat diff;
			int pos = o*cfg.n_scales+s;

			// ========== L channel ==========
			diff = pyr_center_L[o][s]-pyr_surround_L[o][s];
			threshold(diff, on_off_L[pos], 0, 1, THRESH_TOZERO);
			diff *= -1.f;
			threshold(diff, off_on_L[pos], 0, 1, THRESH_TOZERO);

			// ========== a channel ==========
			diff = pyr_center_a[o][s]-pyr_surround_a[o][s];
			threshold(diff, on_off_a[pos], 0, 1, THRESH_TOZERO);
			diff *= -1.f;
			threshold(diff, off_on_a[pos], 0, 1, THRESH_TOZERO);

			// ========== b channel ==========
			diff = pyr_center_b[o][s]-pyr_surround_b[o][s];
			threshold(diff, on_off_b[pos], 0, 1, THRESH_TOZERO);
			diff *= -1.f;
			threshold(diff, off_on_b[pos], 0, 1, THRESH_TOZERO);
		}
	}
}*/

void VOCUS2::orientation(){

	gabor.resize(4);
	for(int i = 0; i < 4; i++) gabor[i].resize(6);

	for(int ori = 0; ori < 4; ori++){
		int filter_size = 2*11*cfg.center_sigma+1;
		float waveLength = 0.6;
		Mat gaborKernel1 = getGaborKernel(Size(filter_size,filter_size), 2*cfg.center_sigma, (ori*M_PI)/4, waveLength, 1.0, CV_PI/2);
		Mat gaborKernel2 = getGaborKernel(Size(filter_size,filter_size), 2*cfg.center_sigma, (ori*M_PI)/4, waveLength, 1.0, 0);
		gaborKernel1.convertTo(gaborKernel1, CV_64FC1);
		gaborKernel2.convertTo(gaborKernel2, CV_64FC1);
		float u = sum(mean(gaborKernel1))[0];
		subtract(gaborKernel1, u, gaborKernel1);
		u = sum(mean(gaborKernel2))[0];
		subtract(gaborKernel2, u, gaborKernel2);

		//For debugging reasons, saving the gabor patches as images
		string dir_gabors = "/home/sevim/catkin_ws/src/vocus2/src/results/gabors";
		double ma, mi;
		int pos = 0;
		for(int o = 2; o <=4; o++){
			for(int s = 2; s <=3; s++){

				Mat& src1 = pyr_center_L[o][0];
				Mat& src2 = pyr_surround_L[o+s][0];
				Mat& dst = gabor[ori][pos];


				Mat out1, out2;
				Mat gabor1, gabor2;

				filter2D(src1, out1, -1, gaborKernel1, Point(-1,-1), 0, BORDER_REPLICATE);
				filter2D(src1, out2, -1, gaborKernel2, Point(-1,-1), 0, BORDER_REPLICATE);

				multiply(out1, out1, out1);
				multiply(out2, out2, out2);
				add(out1, out2, gabor1);
				sqrt(gabor1, gabor1);



				filter2D(src2, out1, -1, gaborKernel1, Point(-1,-1), 0, BORDER_REPLICATE);
				filter2D(src2, out2, -1, gaborKernel2, Point(-1,-1), 0, BORDER_REPLICATE);

				multiply(out1, out1, out1);
				multiply(out2, out2, out2);
				add(out1, out2, gabor2);
				sqrt(gabor2, gabor2);

				resize(gabor2, gabor2, gabor1.size());
				dst = gabor1 - gabor2;
				threshold(dst, dst, 0, 1, THRESH_TOZERO);

				//normalize(dst, dst, 0, 1, NORM_MINMAX);
				//dst = abs(dst);
				//minMaxLoc(dst, &mi, &ma);
				//imwrite(dir_gabors + "/out_" + to_string(ori) + "_octave_"+ to_string(o) + "_scale_" + to_string(s) + ".png", (dst-mi)/(ma-mi)*255.f);


				//minMaxLoc(gabor1, &mi, &ma);
				//imwrite(dir_gabors + "/gabor1_" + to_string(ori) + "_octave_"+ to_string(o) + "_scale_" + to_string(s) + ".png", (gabor1-mi)/(ma-mi)*255.f);

				//minMaxLoc(gabor2, &mi, &ma);
				//imwrite(dir_gabors + "/gabor2_" + to_string(ori) + "_octave_"+ to_string(o) + "_scale_" + to_string(s) + ".png", (gabor2-mi)/(ma-mi)*255.f);

				//minMaxLoc(src, &mi, &ma);
				//imwrite(dir_gabors + "/laplace_" + to_string(o) + "_octave_"+ to_string(s) + ".png", (src-mi)/(ma-mi)*255.f);
				pos++;
			}
		}
	}
}



/*void VOCUS2::orientation(){

	// reserve space
	pyr_laplace.resize(pyr_center_L.size());
	for(int o = 0; o < (int)pyr_center_L.size(); o++){
		pyr_laplace[o].resize(pyr_center_L[o].size());
	}

	gabor.resize(4);
	for(int i = 0; i < 4; i++) gabor[i].resize(pyr_center_L.size()*cfg.n_scales);


	// build all layers of laplace pyramid except the last one
#pragma omp parallel for
	for(int o = 0; o < (int)pyr_center_L.size()-1; o++){
		for(int s = 0; s < (int)pyr_center_L[o].size(); s++){
			Mat& src1 = pyr_center_L[o][s];
			Mat& src2 = pyr_center_L[o+1][s];

			Mat tmp;
			resize(src2, tmp, src1.size(), INTER_NEAREST);

			pyr_laplace[o][s] = src1-tmp;
			threshold(pyr_laplace[o][s], pyr_laplace[o][s], 0, 1, THRESH_TOZERO);
		}
	}

	// copy last layer
	for(int s = 0; s < cfg.n_scales; s++){
		pyr_laplace[pyr_center_L.size()-1][s] = pyr_center_L[pyr_center_L.size()-1][s];
	}



	#pragma omp parallel for
	for(int ori = 0; ori < 4; ori++){
		int filter_size = 2*11*cfg.center_sigma+1;
		float waveLength = 0.6;
		Mat gaborKernel1 = getGaborKernel(Size(filter_size,filter_size), 2*cfg.center_sigma, (ori*M_PI)/4, waveLength, 1.0, CV_PI/2);
		Mat gaborKernel2 = getGaborKernel(Size(filter_size,filter_size), 2*cfg.center_sigma, (ori*M_PI)/4, waveLength, 1.0, 0);
		gaborKernel1.convertTo(gaborKernel1, CV_64FC1);
		gaborKernel2.convertTo(gaborKernel2, CV_64FC1);
		float u = sum(mean(gaborKernel1))[0];
		subtract(gaborKernel1, u, gaborKernel1);
		u = sum(mean(gaborKernel2))[0];
		subtract(gaborKernel2, u, gaborKernel2);

		//std::cout << "Gabor kernel 1, phase 90 summation: " << sum(mean(gaborKernel1))[0] << std::endl;
		//std::cout << "Gabor kernel 2, phase 0 summation: " << sum(mean(gaborKernel2))[0] << std::endl;
		//std::cout << "type: " << gaborKernel1 << std::endl << std::endl;

		//For debugging reasons, saving the gabor patches as images
		string dir_gabors = "/home/sevim/catkin_ws/src/vocus2/src/results/gabors";
		double ma, mi;
		//minMaxLoc(gaborKernel1, &mi, &ma);
		//imwrite(dir_gabors + "/" + to_string(ori) + "_phase_90.png", (gaborKernel1-mi)/(ma-mi)*255.f);

		//minMaxLoc(gaborKernel2, &mi, &ma);
		//imwrite(dir_gabors + "/" + to_string(ori) + "_phase_0.png", (gaborKernel2-mi)/(ma-mi)*255.f);

		for(int o = 0; o < (int)pyr_laplace.size(); o++){
			for(int s = 0; s < cfg.n_scales; s++){


				//float k_sum =  sum(sum(abs(gaborKernel)))[0];
				//gaborKernel /= k_sum;

				int pos = o*cfg.n_scales+s;

				Mat& src = pyr_laplace[o][s];
				//resize(src, src, input.size(), 0, 0, INTER_CUBIC);
				Mat& dst = gabor[ori][pos];

				Mat out1, out2;

				filter2D(src, out1, -1, gaborKernel1, Point(-1,-1), 0, BORDER_REPLICATE);
				filter2D(src, out2, -1, gaborKernel2, Point(-1,-1), 0, BORDER_REPLICATE);

				multiply(out1, out1, out1);
				multiply(out2, out2, out2);
				add(out1, out2, dst);
				sqrt(dst, dst);

				//normalize(dst, dst, 0, 1, NORM_MINMAX);
				//dst = abs(dst);
				//minMaxLoc(dst, &mi, &ma);
				//imwrite(dir_gabors + "/out_" + to_string(ori) + "_octave_"+ to_string(o) + "_scale_" + to_string(s) + ".png", dst*255.f);



				//minMaxLoc(src, &mi, &ma);
				//imwrite(dir_gabors + "/laplace_" + to_string(o) + "_octave_"+ to_string(s) + ".png", (src-mi)/(ma-mi)*255.f);
			}
		}
	}
}
*/


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
	vector<Mat> feature_color1, feature_color2;

	if(cfg.combined_features){
		feature_color1.push_back(fuse(on_off_a, cfg.fuse_feature));
		feature_color1.push_back(fuse(off_on_a, cfg.fuse_feature));
		feature_color1.push_back(fuse(on_off_b, cfg.fuse_feature));
		feature_color1.push_back(fuse(off_on_b, cfg.fuse_feature));
	}
	else{
		feature_color1.push_back(fuse(on_off_a, cfg.fuse_feature));
		feature_color1.push_back(fuse(off_on_a, cfg.fuse_feature));
		feature_color2.push_back(fuse(on_off_b, cfg.fuse_feature));
		feature_color2.push_back(fuse(off_on_b, cfg.fuse_feature));
	}

	vector<Mat> feature_orientation;
	if(cfg.orientation && cfg.combined_features){
		for(int i = 0; i < 4; i++){
			feature_orientation.push_back(fuse(gabor[i], cfg.fuse_feature));
		}
	}

	// conspicuity maps
	vector<Mat> conspicuity_maps;
	conspicuity_maps.push_back(fuse(feature_intensity, cfg.fuse_conspicuity));



	if(cfg.combined_features){
		conspicuity_maps.push_back(fuse(feature_color1, cfg.fuse_conspicuity));
		if(cfg.orientation) conspicuity_maps.push_back(fuse(feature_orientation, cfg.fuse_conspicuity));

	}
	else{
		conspicuity_maps.push_back(fuse(feature_color1, cfg.fuse_conspicuity));
		conspicuity_maps.push_back(fuse(feature_color2, cfg.fuse_conspicuity));
		if(cfg.orientation){
			for(int i = 0; i < 4; i++){
				conspicuity_maps.push_back(fuse(gabor[i], cfg.fuse_feature));
				//compute_weight_by_dilation(conspicuity_maps[conspicuity_maps.size()-1], "gabor" + to_string(i+1) + ".png");
			}

			/*double ma = 0;
			vector<double> maximas(conspicuity_maps.size());
			for(int i = 0; i < conspicuity_maps.size(); i++)
				minMaxLoc(conspicuity_maps[i], nullptr, &maximas[i]);
			ma = *max_element(maximas.begin(), maximas.end());
			for(int i = 0; i < conspicuity_maps.size(); i++){
				normalize(conspicuity_maps[i], conspicuity_maps[i], 0, ma, NORM_MINMAX);
				cout << "WEEEIIIGHHHTT: " << compute_weight_by_dilation(conspicuity_maps[i], "consp_" + to_string(i+1) + ".png") << endl;
			}*/


		}
	}

	// saliency map
	salmap = fuse(conspicuity_maps, cfg.fuse_conspicuity);
	for(int i = 0; i < conspicuity_maps.size(); i++){
		double mi, ma;
		minMaxLoc(conspicuity_maps[i], &mi, &ma);
		imwrite("/home/sevim/catkin_ws/src/vocus2/src/results/conspicuity_" + to_string(i) + ".png", conspicuity_maps[i]*255.f);
	}

	// normalize output to [0,1]
	if(cfg.normalize){
		double mi, ma;
		minMaxLoc(salmap, &mi, &ma);
		salmap = (salmap-mi)/(ma-mi);
	}

	// resize to original image size
	resize(salmap, salmap, input.size(), 0, 0, INTER_CUBIC);

	salmap_ready = true;

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


float VOCUS2::compute_weight_by_dilation(const Mat &src, const std::string &filename){
	Mat tmp;
	threshold(src, tmp, 0.05, 1, THRESH_TOZERO);
	int neighbor=4;
	Mat element = getStructuringElement( MORPH_RECT,
										 Size( 3, 3 ),
										 Point( 1, 1) );
	Mat peak_img = tmp.clone();
	dilate(peak_img,peak_img,element,Point(-1,-1),neighbor);
	peak_img = peak_img - tmp;

	Mat flat_img ;
	erode(tmp,flat_img,element,Point(-1,-1),neighbor);
	flat_img = tmp - flat_img;

	threshold(peak_img,peak_img,0,255,CV_THRESH_BINARY_INV);
	threshold(flat_img,flat_img,0,255,CV_THRESH_BINARY_INV);

	peak_img.convertTo(peak_img, CV_8UC1, 255);
	flat_img.convertTo(flat_img, CV_8UC1, 255);
	peak_img.setTo(Scalar::all(0),flat_img);




	Mat src_copy;
	src.copyTo(src_copy);
	src_copy.convertTo(src_copy, CV_8UC1, 255);
	//peak_img.convertTo(peak_img, CV_32FC1);

	addWeighted(src_copy, 1, peak_img, 1, 0, src_copy);
	imwrite("/home/sevim/catkin_ws/src/vocus2/src/results/weights/" + filename, src_copy);
	//Mat src_copy;
	src.copyTo(src_copy, peak_img);
	float s = sum(src_copy)[0];
	float p = sum(peak_img)[0]/255.f;
	if(p==0) return 0;

	std::cout << "p : " << p << std::endl;
	std::cout << "s : " << s << std::endl;
	double ma;
	minMaxLoc(src, nullptr, &ma);
	std::cout << "ma: " << ma << std::endl;
	float m = s/p;
	std::cout << "m: " << m << std::endl;
	std::cout << "weight: " << ma - m << std::endl;
	//std::cout << src_copy << std::endl;

	return ma-m;
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

//Build multiscale pyramid
vector<vector<Mat> > VOCUS2::build_multiscale_pyr(Mat& mat, float sigma){

	// maximum layer = how often can the image by halfed in the smaller dimension
	// a 320x256 can produce at most 8 layers because 2^8=256
	int max_octaves = min((int)log2(min(mat.rows, mat.cols)), cfg.stop_layer)+1;

	Mat tmp = mat.clone();

	// fast compute unused first layers with one scale per layer
	for(int o = 0; o < cfg.start_layer; o++){
		GaussianBlur(tmp, tmp, Size(), 2.f*sigma, 2.f*sigma, BORDER_REPLICATE);
		resize(tmp, tmp, Size(), 0.5, 0.5, INTER_NEAREST);
	}

	// reserve space
	vector<vector<Mat> > pyr;
	pyr.resize(max_octaves-cfg.start_layer);

	// compute pyramid as it is done in [Lowe2004]
	float sig_prev = 0.f, sig_total = 0.f;

	for(int o = 0; o < max_octaves-cfg.start_layer; o++){
		pyr[o].resize(cfg.n_scales+1);

		// compute an additional scale that is used as the first scale of the next octave
		for(int s = 0; s <= cfg.n_scales; s++){
			Mat& dst = pyr[o][s];

			// if first scale of first used octave => just smooth tmp
			if(o == 0 && s == 0){
				Mat& src = tmp;

				sig_total = pow(2.0, ((double)s/(double)cfg.n_scales))*sigma;
				GaussianBlur(src, dst, Size(), sig_total, sig_total, BORDER_REPLICATE);
				sig_prev = sig_total;
			}

			// if first scale of any other octave => subsample additional scale of previous layer
			else if(o != 0 && s == 0){
				Mat& src = pyr[o-1][cfg.n_scales];
				resize(src, dst, Size(src.cols/2, src.rows/2), 0, 0, INTER_NEAREST);
				sig_prev = sigma;
			}

			// else => smooth an intermediate step
			else{
				sig_total = pow(2.0, ((double)s/(double)cfg.n_scales))*sigma;
				float sig_diff = sqrt(sig_total*sig_total - sig_prev*sig_prev);

				Mat& src = pyr[o][s-1];
				GaussianBlur(src, dst, Size(), sig_diff, sig_diff, BORDER_REPLICATE);
				sig_prev = sig_total;
			}
		}
	}

	// erase all the additional scale of each layer
	for(auto& o : pyr){
		o.erase(o.begin()+cfg.n_scales);
	}

	return pyr;
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

float VOCUS2::compute_uniqueness_weight(Mat& src, float t){
	Mat tmp;
	threshold(src, tmp, 0.05f, 1, THRESH_TOZERO);
	int neighbor=4;
	Mat element = getStructuringElement( MORPH_RECT,
										 Size( 3, 3 ),
										 Point( 1, 1) );
	Mat peak_img = tmp.clone();
	dilate(peak_img,peak_img,element,Point(-1,-1),neighbor);
	peak_img = peak_img - tmp;

	Mat flat_img ;
	erode(tmp,flat_img,element,Point(-1,-1),neighbor);
	flat_img = tmp - flat_img;

	threshold(peak_img,peak_img,0,255,CV_THRESH_BINARY_INV);
	threshold(flat_img,flat_img,0,255,CV_THRESH_BINARY_INV);

	peak_img.convertTo(peak_img, CV_8UC1, 255);
	flat_img.convertTo(flat_img, CV_8UC1, 255);
	peak_img.setTo(Scalar::all(0),flat_img);

	Mat src_copy;
	src.copyTo(src_copy, peak_img);
	float s = sum(src_copy)[0];
	float p = sum(peak_img)[0]/255.f;
	if(p == 0) return 0;

	double ma;
	minMaxLoc(src, nullptr, &ma);
	float m = s/p;

	return ma-m;
}



/*float VOCUS2::compute_uniqueness_weight(Mat& img, float t){
	float s = 0;

	// hold maximal points
	vector<Point> point_maxima;

	// hold maximal blobs
	vector<vector<Point> > blob_maxima;

	CV_Assert(img.channels() == 1);

	// find maximum
	double ma;
	minMaxLoc(img, nullptr, &ma);

	// ignore map if global max is too small
	if(ma < 0.05) return 0.f;

	// ignore values < some portion t of the maximal value
	float thresh = ma*t;
	Mat mask;
	threshold(img, mask, thresh, 1, THRESH_BINARY_INV);
	mask.convertTo(mask, CV_8U);

	// number of maxima
	int n_max = 0;

	// for each image pixel
	for(int r = 0; r < img.rows; r++){
		for(int c = 0; c < img.cols; c++){

			// skip marked pixel
			if(mask.ptr<uchar>(r)[c] != 0) continue;

			float val = img.ptr<float>(r)[c];

			vector<Point> lower, greater, equal;

			// investigate neighborhood for pixel of values
			// greater, lower or equal to the current pixel
			for(int dr = -1; dr <= 1; dr++){
				for(int dc = -1; dc <= 1; dc++){
					// skip current pixel
					if(dr == 0 && dc == 0) continue;

					// skip out of bound pixels
					if(r+dr < 0 || r+dr >= img.rows) continue;
					if(c+dc < 0 || c+dc >= img.cols) continue;

					float tmp = img.ptr<float>(r+dr)[c+dc];
					Point p = Point(c+dc, r+dr);

					if(tmp < val) lower.push_back(p);
					else if(tmp > val) greater.push_back(p);
					else equal.push_back(p);
				}
			}

			// case 1: isolated point
			if(equal.size() == 0){

				// current point is done
				mask.ptr<uchar>(r)[c] = 1;

				// all smaller neighbours are definitive no maxima
				for(Point& p : lower) mask.ptr<uchar>(p.y)[p.x] = 1;

				// if no greater neighbours => maximum
				if(greater.size() == 0){
					// add as maximum
					point_maxima.push_back(Point(c,r));
					s+=val;
					n_max++;
				}
			}

			// case 2: blob
			else{
				Mat considered = Mat::zeros(img.size(), CV_8U);

				// mark all pixel as considered
				for(Point& p : lower) considered.ptr<uchar>(p.y)[p.x] = 1;
				for(Point& p : equal) considered.ptr<uchar>(p.y)[p.x] = 1;
				for(Point& p : greater) considered.ptr<uchar>(p.y)[p.x] = 1;
				considered.ptr<uchar>(r)[c] = 1;

				// extent point to blob
				int pos = 0;
				while(pos < (int)equal.size()){
					int nr = equal[pos].y;
					int nc = equal[pos].x;

					for(int dr = -1; dr <= 1; dr++){
						for(int dc = -1; dc <= 1; dc++){
							// skip current pixel
							if(dr == 0 && dc == 0) continue;

							// skip out of bound pixels
							if(nr+dr < 0 || nr+dr >= img.rows) continue;
							if(nc+dc < 0 || nc+dc >= img.cols) continue;

							// skip considered pixels
							if(considered.ptr<uchar>(nr+dr)[nc+dc] == 1) continue;

							float tmp = img.ptr<float>(nr+dr)[nc+dc];
							Point p = Point(nc+dc, nr+dr);

							if(tmp < val) lower.push_back(p);
							else if(tmp > val) greater.push_back(p);
							else equal.push_back(p);

							considered.ptr<uchar>(p.y)[p.x] = 1;
						}
					}
					pos++;
				}

				// mark all lower neighbours (definitive no maxima)
				for(Point& p : lower) mask.ptr<uchar>(p.y)[p.x] = 1.f;

				// mark all blob pixels (maxima)
				equal.push_back(Point(c,r));
				for(Point& p : equal) mask.ptr<uchar>(p.y)[p.x] = 1.f;

				// case 2.1: all neighbours are lower
				if(greater.size() == 0){
					blob_maxima.push_back(equal);
					s+=val;
					n_max++;
				}
			}
		}
	}

	if(n_max == 0) return 0.f;
	else if(s/n_max>1) return 0.f;
	else{
		//std::cout << "weight is: " << 1 - s/n_max << std::endl;
		//Mat tmp;
		//resize(img, tmp, input.size(), 0, 0, INTER_CUBIC);
		//double ma;
		//minMaxLoc(img, nullptr, &ma);
		//imwrite("/home/sevim/catkin_ws/src/vocus2/src/results/weights/map_" + to_string(ma - s/n_max) + "_"+ to_string(ma)+"_"+ to_string(s/n_max) + ".png", tmp*255.f);
		return ma - s/n_max;
		//return 1/sqrt(n_max);
	}
	/*else  {
		//Mat tmp;
		//resize(img, tmp, input.size(), 0, 0, INTER_CUBIC);
		//imwrite("/home/sevim/catkin_ws/src/vocus2/src/results/weights/map_" + to_string(n_max) + ".png", tmp*255.f);
		return 1/sqrt(n_max);
	}*/
//}



//Fuse maps using operation
Mat VOCUS2::fuse(vector<Mat> &maps, FusionOperation op, bool norm){

	// resulting map that is returned
	Mat fused = Mat::zeros(maps[0].size(), CV_32F);
	int n_maps = maps.size();	// no. of maps to fuse
	vector<Mat> resized;		// temp. array to hold the resized maps
	resized.resize(n_maps);		// reserve space (needed to use openmp for parallel resizing)
	if(norm){
		double ma = 0;
		vector<double> maximas(maps.size());
		for(int i = 0; i < maps.size(); i++)
			minMaxLoc(maps[i], nullptr, &maximas[i]);

		ma = *max_element(maximas.begin(), maximas.end());
		for(int i = 0; i < maximas.size(); i++){
			/*maximas[i] /= ma;
			if(maximas[i]>0)
				maps[i] /=maximas[i];*/
			normalize(maps[i], maps[i], 0, ma, NORM_MINMAX);
		}

	}

	// ========== ARTIMETIC MEAN ==========

	if(op == ARITHMETIC_MEAN){
#pragma omp parallel for schedule(dynamic, 1)
		for(int i = 0; i < n_maps; i++){
			if(fused.size() != maps[i].size()){
				resize(maps[i], resized[i], fused.size(), 0, 0, INTER_CUBIC);
			}
			else{
				resized[i] = maps[i];
			}
		}

		for(int i = 0; i < n_maps; i++){
			cv::add(fused, resized[i], fused, Mat(), CV_32F);
		}

		fused /= (float)n_maps;
	}

	// ========== MAX ==========

	else if(op == MAX){

#pragma omp parallel for schedule(dynamic, 1)
		for(int i = 0; i < n_maps; i++){
			resize(maps[i], resized[i], fused.size(), 0, 0, INTER_CUBIC);
		}

		for(int i = 0; i < n_maps; i++){
#pragma omp parallel for
			for(int r = 0; r < fused.rows; r++){
				float* row_tmp = resized[i].ptr<float>(r);
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
			//std::cout << "Uniqueness weight: " << weight[i] << std::endl;
			if(weight[i] > 0){
				resize(maps[i]*weight[i], resized[i], fused.size(), 0, 0, INTER_CUBIC);
			}
		}

		float sum_weights = 0;

		for(int i = 0; i < n_maps; i++){
			if(weight[i] > 0){
				sum_weights += weight[i];
				cv::add(fused, resized[i], fused, Mat(), CV_32F);
			}
		}

		if(sum_weights > 0) fused /= sum_weights;
	}

	return fused;
}
vector<Mat> VOCUS2::prepare_input(const Mat& img){

	CV_Assert(img.channels() == 3);
	vector<Mat> planes;
	planes.resize(3);

	// convert colorspace and split to single planes
	if(cfg.c_space == LAB){
		Mat converted;
		// convert colorspace (important: before conversion to float to keep range [0:255])
    cvtColor(img, converted, CV_BGR2Lab);
		// convert to float
    converted.convertTo(converted, CV_32FC3);
		// scale down to range [0:1]
		converted /= 255.f;
		split(converted, planes);
	}

	// opponent color as in CoDi (todo: maybe faster with openmp)
	else if(cfg.c_space == OPPONENT_CODI){
		Mat converted;
		img.convertTo(converted, CV_32FC3);

		vector<Mat> planes_bgr;
		split(converted, planes_bgr);

		planes[0] = planes_bgr[0] + planes_bgr[1] + planes_bgr[2];
		planes[0] /= 3*255.f;

		planes[1] = planes_bgr[2] - planes_bgr[1];
		planes[1] /= 255.f;

		planes[2] = planes_bgr[0] - (planes_bgr[1] + planes_bgr[2])/2.f;
		planes[2] /= 255.f;
	}
	else if(cfg.c_space == OPPONENT){
		Mat converted;
		img.convertTo(converted, CV_32FC3);

		vector<Mat> planes_bgr;
		split(converted, planes_bgr);

		planes[0] = planes_bgr[0] + planes_bgr[1] + planes_bgr[2];
		planes[0] /= 3*255.f;

		planes[1] = planes_bgr[2] - planes_bgr[1] + 255.f;
		planes[1] /= 2*255.f;

		planes[2] = planes_bgr[0] - (planes_bgr[1] + planes_bgr[2])/2.f + 255.f;
		planes[2] /= 2*255.f;
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
