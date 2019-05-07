#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/dir_nav.h>
#include <dlib/opencv/to_open_cv.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/image_transforms.h>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include <iostream>

using namespace dlib;
using namespace std;

/** Global variables */
string descriptor = "./shape_predictor_68_face_landmarks.dat";
bool debug = true;

cv::Mat homography_warp(cv::Mat src,cv::Mat H);
cv::Mat loadImage(cv::String path);
std::vector<cv::Point2f> getLandmark(full_object_detection obj);
cv::Mat drawLandmark(cv::Mat mat_1, std::vector<cv::Point2f> landmark_1, cv::Mat mat_2, std::vector<cv::Point2f> landmark_2);
cv::Rect dlibRectangleToOpenCV(dlib::rectangle r);

void readme();
// ----------------------------------------------------------------------------------------

int main(int argc, char** argv)
{  
    // 2 pictures must be given
    if( argc != 3 )
    { readme(); return -1; }
    /* Load Face Detector */
    // For face bounding
    frontal_face_detector detector = get_frontal_face_detector();
    /* Load Face Descriptor */
    // For extract 68 points face landmarks
    shape_predictor sp;
    deserialize(descriptor) >> sp;

    /* New plan */
    /* Use DLIB for get descriptor */

    /* Load images */
    // ./detector img_1 img_2
    array2d<rgb_pixel> img_1, img_2;
	// array2d<rgb_pixel> i1,i2;
    load_image(img_1, argv[1]);
    load_image(img_2, argv[2]);
	// resize_image(i1, img_1, interpolate_billinear());
	// resize_image(i2, img_2, interpolate_billinear());

    /* Process images 1 */
    /* Detect Face */
    cout << "processing image " << argv[1] << endl;
    std::vector<rectangle> dets_1 = detector(img_1);
    cout << "--Number of faces detected: " << dets_1.size() << endl;
    /* Descriptor */
    std::vector<full_object_detection> faces_1;
    for(int i = 0; i < dets_1.size(); i++){
        full_object_detection face = sp(img_1, dets_1[i]);
        cout << "----number of landmark of face " << to_string(i) << ": "<< face.num_parts() << endl;
        faces_1.push_back(face);
		for(int k = 0; k < face.num_parts(); k++){
			draw_solid_circle(img_1, face.part(k), 3,rgb_pixel( 255, 255, 0 ) );
		}
    }
    if(debug) save_jpeg(img_1, "faces_1.jpg" );

    /* Process images 2 */
    /* Detect Face */
    cout << "processing image " << argv[2] << endl;
    std::vector<rectangle> dets_2 = detector(img_2);
    cout << "--Number of faces detected: " << dets_2.size() << endl;
    /* Descriptor */
    std::vector<full_object_detection> faces_2;
    for(int i = 0; i < dets_2.size(); i++){
        full_object_detection face = sp(img_2, dets_2[i]);
        cout << "----number of landmark of face " << to_string(i) << ": "<< face.num_parts() << endl;
        faces_2.push_back(face);
		for(int k = 0; k < face.num_parts(); k++){
			draw_solid_circle(img_2, face.part(k), 3,rgb_pixel( 255, 255, 0 ) );
		}
    }
    if(debug) save_jpeg(img_2, "faces_2.jpg" );


    /* For each found faces in img_1  */
    for(int i = 0; i < dets_1.size(); i++){
    	std::vector<cv::Point2f> landmark_1 = getLandmark(faces_1[i]);
    	/* compare to each found faces in img_2 */
    	for(int j = 0; j < dets_2.size(); j++){
	    	std::vector<cv::Point2f> landmark_2 = getLandmark(faces_2[j]);

	    	cv::String pairName = to_string(i) + "_" + to_string(j);
	    	cout << "Compare Face: " << pairName << endl;

    		/* Load picture again for drawing */
			cv::Mat mat_1 = loadImage(argv[1]);
			cv::Mat mat_2 = loadImage(argv[2]);
			cv::Mat mat_c = drawLandmark(mat_1, landmark_1, mat_2, landmark_2);
			cv::imwrite(pairName + "_match.jpg", mat_c);
			/* find Homography */
			cv::Mat H = cv::findHomography( landmark_1, landmark_2 );
			cout << "H = "<< endl << " "  << H << endl << endl;
			
			/* Compare 2 faces */
			cv::Mat face_1 = mat_1(dlibRectangleToOpenCV(faces_1[i].get_rect()));
			cv::Mat face_2 = mat_2(dlibRectangleToOpenCV(faces_2[j].get_rect()));
			cout << "WARP!!" << endl;
			// cv::Mat mat_1_w, mat_c_w;
			// warpPerspective( mat_1, mat_1_w, H, mat_1.size());
			// copyMakeBorder( mat_1_w, mat_1_w, 0, mat_2.rows - mat_1_w.rows, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0,0,0) );
			// cv::hconcat(mat_1_w, mat_2, mat_c_w);
			// cv::imwrite(pairName + "_warp_1.jpg", mat_c_w);
			cout << "CONCAT!!" << endl;
			// cout << "FACE_1 col:" << face_1.cols << " row:" << face_1.rows << endl;
			// cout << "FACE_2 col:" << face_2.cols << " row:" << face_2.rows << endl;
			face_1 = homography_warp(face_1, H);
			if(face_1.rows < face_2.rows){
				copyMakeBorder( face_1, face_1, 0, face_2.rows - face_1.rows, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0,0,0) );
			}
			if(face_2.rows < face_1.rows){
				copyMakeBorder( face_2, face_2, 0, face_1.rows - face_2.rows, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0,0,0) );
			}
			cv::Mat face_c;
			cv::hconcat(face_1, face_2, face_c);
			cv::imwrite(pairName + "_warp_1.jpg", face_c);

			/* find reprojection error */
			// cout << "FACE_1 col:" << face_1 << " row:" << face_1.rows << endl;
    	}
    }
}

cv::Mat loadImage(cv::String path){
	cv::Mat mat = cv::imread( path );
	// resize(mat,mat,cv::Size(1920,1080));//resize image
	return mat;
}

cv::Rect dlibRectangleToOpenCV(dlib::rectangle r){
  return cv::Rect(cv::Point2i(r.left(), r.top()), cv::Point2i(r.right() + 1, r.bottom() + 1));
}

std::vector<cv::Point2f> getLandmark(full_object_detection obj){
	std::vector<cv::Point2f> landmark(obj.num_parts());
	for(int k = 0; k < obj.num_parts(); k++){
		// cout << "get landmark:" + to_string(k) << endl;
		landmark[k] = cv::Point2f(obj.part(k).x(), obj.part(k).y());
	}
	return landmark;
}

cv::Mat drawLandmark(cv::Mat mat_1, std::vector<cv::Point2f> landmark_1, cv::Mat mat_2, std::vector<cv::Point2f> landmark_2){
	cv::Mat mat_c;
    if(mat_1.rows < mat_2.rows){
        copyMakeBorder( mat_1, mat_1, 0, mat_2.rows - mat_1.rows, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0,0,0) );
    }
    if(mat_2.rows < mat_1.rows){
        copyMakeBorder( mat_2, mat_2, 0, mat_1.rows - mat_2.rows, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0,0,0) );
    }
	// copyMakeBorder( mat_1, mat_1, 0, mat_2.rows - mat_1.rows, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0,0,0) );
	cv::hconcat(mat_1, mat_2, mat_c);
	for(int i = 0 ; i < landmark_1.size(); i++){
		/* draw point for face 1 */
		circle(mat_c, landmark_1[i], 3, cv::Scalar(255, 255, 0), -1);
		circle(mat_c, landmark_2[i] + cv::Point2f( mat_1.cols, 0), 3, cv::Scalar(255, 255, 0), -1);
		cv::line(mat_c, landmark_1[i], landmark_2[i] + cv::Point2f( mat_1.cols, 0), cv::Scalar(0, 255, 0) );
	}
	return mat_c;
}

/* @function readme */
void readme()
{ cout << " Usage: ./detector shape_predictor_68_face_landmarks.dat ../me/attack5.jpg " << endl; }





/* https://stackoverflow.com/questions/22220253/cvwarpperspective-only-shows-part-of-warped-image */
// Convert a vector of non-homogeneous 2D points to a vector of homogenehous 2D points.
void to_homogeneous(const std::vector< cv::Point2f >& non_homogeneous, std::vector< cv::Point3f >& homogeneous)
{
    homogeneous.resize(non_homogeneous.size());
    for (size_t i = 0; i < non_homogeneous.size(); i++) {
        homogeneous[i].x = non_homogeneous[i].x;
        homogeneous[i].y = non_homogeneous[i].y;
        homogeneous[i].z = 1.0;
    }
}

// Convert a vector of homogeneous 2D points to a vector of non-homogenehous 2D points.
void from_homogeneous(const std::vector< cv::Point3f >& homogeneous, std::vector< cv::Point2f >& non_homogeneous)
{
    non_homogeneous.resize(homogeneous.size());
    for (size_t i = 0; i < non_homogeneous.size(); i++) {
        non_homogeneous[i].x = homogeneous[i].x / homogeneous[i].z;
        non_homogeneous[i].y = homogeneous[i].y / homogeneous[i].z;
    }
}

// Transform a vector of 2D non-homogeneous points via an homography.
std::vector<cv::Point2f> transform_via_homography(const std::vector<cv::Point2f>& points, const cv::Matx33f& homography)
{
    std::vector<cv::Point3f> ph;
    to_homogeneous(points, ph);
    for (size_t i = 0; i < ph.size(); i++) {
        ph[i] = homography*ph[i];
    }
    std::vector<cv::Point2f> r;
    from_homogeneous(ph, r);
    return r;
}

// Find the bounding box of a vector of 2D non-homogeneous points.
cv::Rect_<float> bounding_box(const std::vector<cv::Point2f>& p)
{
    cv::Rect_<float> r;
    float x_min = std::min_element(p.begin(), p.end(), [](const cv::Point2f& lhs, const cv::Point2f& rhs) {return lhs.x < rhs.x; })->x;
    float x_max = std::max_element(p.begin(), p.end(), [](const cv::Point2f& lhs, const cv::Point2f& rhs) {return lhs.x < rhs.x; })->x;
    float y_min = std::min_element(p.begin(), p.end(), [](const cv::Point2f& lhs, const cv::Point2f& rhs) {return lhs.y < rhs.y; })->y;
    float y_max = std::max_element(p.begin(), p.end(), [](const cv::Point2f& lhs, const cv::Point2f& rhs) {return lhs.y < rhs.y; })->y;
    return cv::Rect_<float>(x_min, y_min, x_max - x_min, y_max - y_min);
}

// Warp the image src into the image dst through the homography H.
// The resulting dst image contains the entire warped image, this
// behaviour is the same of Octave's imperspectivewarp (in the 'image'
// package) behaviour when the argument bbox is equal to 'loose'.
// See http://octave.sourceforge.net/image/function/imperspectivewarp.html
cv::Mat homography_warp(cv::Mat src, cv::Mat H)
{
    std::vector< cv::Point2f > corners;
    corners.push_back(cv::Point2f(0, 0));
    corners.push_back(cv::Point2f(src.cols, 0));
    corners.push_back(cv::Point2f(0, src.rows));
    corners.push_back(cv::Point2f(src.cols, src.rows));

    std::vector< cv::Point2f > projected = transform_via_homography(corners, H);
    cv::Rect_<float> bb = bounding_box(projected);

    cv::Mat_<double> translation = (cv::Mat_<double>(3, 3) << 1, 0, -bb.tl().x, 0, 1, -bb.tl().y, 0, 0, 1);
    cv::Mat dst;
    cv::warpPerspective(src, dst, translation*H, bb.size());
    return dst;
}