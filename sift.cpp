#include <stdio.h>
#include <sstream>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>

// #include <opencv2/cudaoptflow.hpp>

#include <opencv2/optflow.hpp>


#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>



#include "helpers/OpenCVHelper.hpp"
#include "helpers/optical_flow_helpers.hpp"
#include "helpers/util.hpp"

using namespace cv;

int main(int argc, char* argv[] )
{
    using namespace cv;

    if (cmd_option_exists(argv, argv+argc, "-h")
        || argc < 3){
        std::cout << "LTP calculation app\n\n"
                  << "./sift [image1] [image2] \n\n"
                  << "flags:\n"  
                  << "-o: output file path\n" 
                  << std::endl; 

        return 0;
    } 


    std::string img1path = argv[1];
    std::string img2path = argv[2];

    std::string outpath = "";
    if (cmd_option_exists(argv, argv+argc, "-o")){
        outpath = get_cmd_option(argv, argv+argc, "-o");
    } 



    using namespace cv;

    const int     nfeatures         = 100; // 0 = no limit
    const int     nOctaveLayers     = 5;
    const double  contrastThreshold = 0.04;
    const double  edgeThreshold     = 10;
    const double  sigma             = 1.0;

    auto sift =  cv::xfeatures2d::SIFT::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);

    const int normType = NORM_L2; // best for SIFT. ORB and other binary descriptors should be compared with NORM_HAMMING2
    const bool crossCheck = false; // check in both directions for more reliable matches
    
    auto matcher = BFMatcher::create(normType, crossCheck); 

    std::vector<std::vector<KeyPoint>> keypoints (2);
    Mat descriptors[2];

    

    const double MATCH_RATIO = 0.75;
    
    std::cout << "Loading images for matching:\n"  << std::endl;

    Mat image1 = imread(img1path, IMREAD_GRAYSCALE);
    Mat image2 = imread(img2path, IMREAD_GRAYSCALE);

    // detect keypoints in each image
    sift->detect(image1, keypoints[0]);
    sift->detect(image2, keypoints[1]);
    // compute descriptors
    sift->compute( image1, keypoints[0], descriptors[0]);
    sift->compute( image2, keypoints[1], descriptors[1]);
    // find strong matches between keypoints

    const bool RATIO_TEST = false;

    std::vector<DMatch> matches;

    if (RATIO_TEST){

        // alternative: knn matches for multiple matches
        std::vector<std::vector<DMatch>> knn_matches;
        const int k = 2;
        matcher->knnMatch(descriptors[0], descriptors[1],knn_matches, k);

        for (const auto& match_set : knn_matches){
            if ( match_set[0].distance < MATCH_RATIO * match_set[1].distance ){
                matches.push_back(match_set[0]);
            }
        }

    } else {

        matcher->match( descriptors[0], descriptors[1], matches); // alternative: knn matches for multiple matches
    }


    Mat out_img_matches;
    drawMatches( image1, keypoints[0], 
                 image2, keypoints[1], 
                 matches, 
                 out_img_matches, 
                 Scalar(0,255,0), // match colour
                 Scalar(0,0,255) // no-match colour
                 // std::vector<char>(), 
                 // DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS 
                 );

    if ("" != outpath) {
    	imwrite(outpath, out_img_matches);
    }

    imshow("matches", out_img_matches);
    waitKey(0);


    return 0;
}