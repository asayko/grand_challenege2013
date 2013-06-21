#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>

using std::vector;
using std::cout;
using std::endl;
using std::ofstream;
using std::string;
using std::atoi;

// g++ imgs2sifts.cpp `pkg-config opencv --cflags --libs`

int main(int argc, char *argv[]) {

    if (argc < 3) {
        cout << string("Usage: ") + argv[0] + " MODE (0-view sift interest point, 1-save sift decriptors, 2-view and save) FILENAME.jpeg" << endl;
        exit(0);
    }

    size_t mode = atoi(argv[1]);
    cv::Mat image = cv::imread(argv[2]);

    // Create smart pointer for SIFT feature detector.
    cv::Ptr<cv::FeatureDetector> featureDetector = new cv::SiftFeatureDetector(0, 3, 0.08, 5, 1.4);
    vector<cv::KeyPoint> keypoints;

    // Detect the keypoints
    featureDetector->detect(image, keypoints);
    cout << keypoints.size() << " keypoints detected." << endl;


    //Similarly, we create a smart pointer to the SIFT extractor.
    cv::Ptr<cv::SiftDescriptorExtractor> featureExtractor = new cv::SiftDescriptorExtractor();


    // Compute the 128 dimension SIFT descriptor at each keypoint.
    // Each row in "descriptors" correspond to the SIFT descriptor for each keypoint
    cv::Mat descriptors;
    featureExtractor->compute(image, keypoints, descriptors);

    cout << descriptors.rows << " descriptors extracted." << endl;

    // Save the extracted descriptors if needed
    if (mode == 1 || mode == 2) {
        cout << "Saving descriptors to " << string(argv[2]) << ".sifts" << endl;
        ofstream fout((string(argv[2]) + ".sifts").c_str());
        for (size_t rowi = 0; rowi < descriptors.rows; ++rowi) {

            cv::Mat tmp;
            cv::normalize(descriptors.row(rowi), tmp);
            tmp.copyTo(descriptors.row(rowi));

            for (size_t coli = 0; coli < descriptors.cols; ++coli) {
                fout << descriptors.at<float>(rowi, coli) << "\t";
            }
            fout << endl;
        }
        cout << "Saved." << endl;
    }


    // If you would like to draw the detected keypoint just to check
    if (mode == 0 || mode == 2) {
        cout << "Plotting the "<< string(argv[2]) << " with sift keypoints marked. Press q to close the window." << endl;
        cv::Mat outputImage;
        cv::Scalar keypointColor = cv::Scalar(255, 0, 0);     // Blue keypoints.
        cv::drawKeypoints(image, keypoints, outputImage, keypointColor, cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        cv::namedWindow(string(argv[1]) + " with marked SIFT interest points.");
        imshow(string(argv[1]) + " with marked SIFT interest points.", outputImage);

        char c = ' ';
        while ((c = cv::waitKey(0)) != 'q');  // Keep window there until user presses 'q' to quit.
    }

    return 0;
}
