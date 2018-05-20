/*****************************************************************************
*
* VOCUS2.h file for the saliency program VOCUS2.
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

#ifndef VOCUS2_CUDA_EXT_H_
#define VOCUS2_CUDA_EXT_H_

#include <opencv2/core/core.hpp>

#include <string>
#include <fstream>

#include <boost/archive/xml_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>

#include <boost/serialization/nvp.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/extended_type_info.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/list.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/assume_abstract.hpp>

#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/cuda.hpp"
//#include "opencv2/cudaarithm.hpp"

using namespace std;
using namespace cv;
using namespace cv::cuda;


// different colorspaces

enum ColorSpace{
	// CIELab
	LAB = 0,
	OPPONENT_CODI = 1, // like in Klein/Frintrop DAGM 2012
	OPPONENT = 2, 	// like above but shifted and scaled to [0,1]
	// splitted RG and BY channels
	ITTI = 3
};

enum FeatureChannels{
    ON_OFF_L = 0,
    OFF_ON_L = 1,
    ON_OFF_A = 2,
    OFF_ON_A = 3,
    ON_OFF_B = 4,
    OFF_ON_B = 5,
    GABOR = 6
};

// fusing operation to build the feature, conspicuity and saliency map(s)
enum FusionOperation{
	ARITHMETIC_MEAN = 0,
	MAX = 1,
	// uniqueness weight as in
	// Simone Frintrop: VOCUS: A Visual Attention System for Object Detection and Goal-directed Search, PhD thesis 2005
	UNIQUENESS_WEIGHT = 2,
};

// pyramid structure
enum PyrStructure{
	// two independent pyramids
	CLASSIC = 0,
	// two independent pyramids derived from a base pyramid
	CODI = 1,
	// surround pyramid derived from center pyramid
	NEW = 2,
	// single pyramid (iNVT)
	SINGLE = 3
};

// class containing all parameters for the main class
class VOCUS2_Cfg{
public:
	// default constructor, default parameters
	VOCUS2_Cfg(){
        c_space = OPPONENT_CODI;
        fuse_feature = UNIQUENESS_WEIGHT;
        fuse_conspicuity = ARITHMETIC_MEAN;
				start_layer = 0;
        stop_layer = 8;
        center_sigma = 1;
        surround_sigma = 1;
        n_scales = 9;
        normalize = true;
        pyr_struct = CLASSIC;
        orientation = true;
        combined_features = true;
	};

	// constuctor for a given config file
	VOCUS2_Cfg(string f_name){
		load(f_name);
	}

	virtual ~VOCUS2_Cfg(){};


	ColorSpace c_space;
	FusionOperation fuse_feature, fuse_conspicuity;
	PyrStructure pyr_struct;

	int start_layer, stop_layer, n_scales;
	float center_sigma, surround_sigma;

	bool normalize, orientation, combined_features;

	// load xml file
	bool load(string f_name){
        std::ifstream conf_file(f_name.c_str());
		if (conf_file.good()) {
			boost::archive::xml_iarchive ia(conf_file);
			ia >> boost::serialization::make_nvp("VOCUS2_Cfg", *this);
			conf_file.close();
			return true;
		}
		else cout << "Config file: " << f_name << " not found. Using defaults." << endl;
		return false;
	}

	// wite to xml file
	bool save(string f_name){
        std::ofstream conf_file(f_name.c_str());
		if (conf_file.good()) {
			boost::archive::xml_oarchive oa(conf_file);
			oa << boost::serialization::make_nvp("VOCUS2_Cfg", *this);
			conf_file.close();
			return true;
		}
		return false;
	}

private:
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version){
    	ar & BOOST_SERIALIZATION_NVP(c_space);
    	ar & BOOST_SERIALIZATION_NVP(fuse_feature);
    	ar & BOOST_SERIALIZATION_NVP(fuse_conspicuity);
		ar & BOOST_SERIALIZATION_NVP(pyr_struct);
    	ar & BOOST_SERIALIZATION_NVP(start_layer);
    	ar & BOOST_SERIALIZATION_NVP(stop_layer);
    	ar & BOOST_SERIALIZATION_NVP(center_sigma);
    	ar & BOOST_SERIALIZATION_NVP(surround_sigma);
    	ar & BOOST_SERIALIZATION_NVP(n_scales);
		ar & BOOST_SERIALIZATION_NVP(normalize);
    }
};

class VOCUS2_cuda_ext {
public:
	VOCUS2_cuda_ext();
	VOCUS2_cuda_ext(const VOCUS2_Cfg& cfg);
	virtual ~VOCUS2_cuda_ext();

	void setCfg(const VOCUS2_Cfg& cfg);
	// computes center surround contrast on the pyramids
	// does not produce the final saliency map
	// has to be called first
	void process(const Mat& image);

	// add a center bias to the final saliency map
	Mat add_center_bias(float lambda);

	// computes the final saliency map given that process() was called
  	Mat get_salmap();

	// computes a saliency map for each layer of the pyramid
	vector<Mat> get_splitted_salmap();

	// write all intermediate results to the given directory
	void write_out(string dir);

    std::vector<cv::Mat> getFeatureChannel(FeatureChannels name);

private:
	VOCUS2_Cfg cfg;
	Mat input;

	Mat salmap;
	vector<Mat> salmap_splitted, planes;

	// vectors to hold contrast pyramids as arrays
	vector<Mat> on_off_L, off_on_L;
	vector<Mat> on_off_a, off_on_a;
	vector<Mat> on_off_b, off_on_b;
    vector<Mat> on_off_gabor0, off_on_gabor0;
    vector<Mat> on_off_gabor45, off_on_gabor45;
    vector<Mat> on_off_gabor90, off_on_gabor90;
    vector<Mat> on_off_gabor135, off_on_gabor135;

	// vector to hold the gabor pyramids

	// vectors to hold center and surround gaussian pyramids
    vector<Mat> pyr_center_L, pyr_surround_L;
    vector<Mat> pyr_center_a, pyr_surround_a;
    vector<Mat> pyr_center_b, pyr_surround_b;



	// vector to hold the edge (laplace) pyramid
	vector<vector<Mat> > pyr_laplace;

	bool salmap_ready, splitted_ready, processed;

	// process image wrt. the desired pyramid structure
	void gaborFilterImages(const Mat &base, Mat &out, int orientation, float sigma, int scale);
	void pyramid_classic(const Mat& image);
	void pyramid_new(const Mat& image);
	void pyramid_codi(const Mat& image);
	// void pyramid_itti(const Mat& image);

	// converts the image to the destination colorspace
	// and splits the color channels
	vector<Mat> prepare_input(const Mat& img);

	// clear all datastructures from previous results
	void clear();

	// build a multi scale representation based on [Lowe2004]
    vector<Mat> build_multiscale_pyr(Mat& img, float sigma = 1.f);

	// combines a vector of Mats into a single mat
	Mat fuse(vector<Mat> mat_array, FusionOperation op);

	// computes the center surround contrast
	// uses pyr_center_L
	void center_surround_diff();
    void orientationWithCenterSurroundDiff();

	// computes the uniqueness of a map by counting the local maxima
	float compute_uniqueness_weight(Mat& map, float t);


	// void mark_equal_neighbours(int r, int c, float value, Mat& map, Mat& marked);

};


#endif /* VOCUS2_H_ */
