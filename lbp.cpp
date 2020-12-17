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



#include "helpers/OpenCVHelper.hpp"
#include "helpers/optical_flow_helpers.hpp"
#include "helpers/util.hpp"



using namespace cv;

std::vector<float> load_depth_image (const std::string& path, 
                                    const uint32_t width,
                                    const uint32_t height

                                     ) {

    std::vector<float> data (width*height);
    std::ifstream infile (path, std::ios::binary);
    infile.read(reinterpret_cast<char*> (data.data()), sizeof(float) * width * height);
    infile.close();
    return data;
}

void normalise_depth_image (std::vector<float>& depth_data, const float min_depth, const float max_depth){

    std::transform(depth_data.begin(), depth_data.end(), depth_data.begin(), [&] (const float d) {
        return (d - min_depth)/(max_depth - min_depth);
    });
}

int main(int argc, char* argv[] )
{
    using namespace cv;

    if (cmd_option_exists(argv, argv+argc, "-h")
        || argc < 4){
        std::cout << "LTP calculation app\n\n"
                  << "./ltp [depth image] [width] [height] \n\n"
                  << "flags:\n"  
                  // << "-o: output file path\n" 
                  << std::endl; 

        return 0;
    } 


    std::string img1path = argv[1];
    const uint32_t width = atoi(argv[2]);
    const uint32_t height = atoi(argv[3]);
    
    // std::string outpath = "";
    // if (cmd_option_exists(argv, argv+argc, "-o")){
    //     outpath = get_cmd_option(argv, argv+argc, "-o");
    // } 

    const float min_depth = 0.5f;
    const float max_depth = 3.f;
    
    // load depth image
    std::vector<float> depth = load_depth_image (img1path, width, height);
    normalise_depth_image (depth, min_depth, max_depth);


    Mat depth_image (height, width, CV_32FC1, depth.data());

    imshow("depth", depth_image);
    waitKey(0);

    std::vector<uint8_t> lbp (width * height, 0);



    // mapping from loop order to descriptor order
    // from: stackoverflow.com/questions/27191047/calculating-the-local-ternary-pattern-of-an-image/28535964?noredirect=1#comment45386491_28535964
    // the final pattern is reading the bit pattern starting from the east location 
    // with respect to the centre (row 2, column 3), then going around counter-clockwise. 
    std::vector<uint8_t> desc_order = {3,2,1,4,0,5,6,7};

    for (uint32_t y = 1; y < height-1; ++y){
        for (uint32_t x = 1; x < width-1; ++x)
        {

            const uint32_t idx =  x + y * width;
            const float val = depth [idx];
            // std::vector<float> nbr_values (8);



            uint8_t count = 0;
            for (int ky = -1; ky < 2; ++ky){
                for (int kx = -1; kx < 2; ++kx){
                    
                    if (ky == 0 && kx == 0) continue;
                    const float nbr_value = depth [ (x+kx) + (y+ky) * width ];

                    if (nbr_value > val){
                        lbp [idx] |= (1<< desc_order[count]);
                    }

                    ++count;
                }
            }
        }
    }


    Mat lbp_img (height, width, CV_8UC1, lbp.data());

    imshow("lbp", lbp_img);
    waitKey(0);

    return 0;
}