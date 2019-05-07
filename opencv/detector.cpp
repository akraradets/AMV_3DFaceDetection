#include <stdio.h>
#include <iostream>
#include <tuple>
#include <string>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/objdetect.hpp"
using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

/** Global variables */
String face_cascade_name = "../haarcascade_frontalface_alt.xml";
CascadeClassifier face_cascade;
RNG rng(12345);

/** Function Headers */
void drawRectagleOnFace(Mat img, vector<Rect> faces);
vector<Rect> findFaces( Mat frame );
tuple<Mat, Mat> checkHomo(Mat img_object, Mat img_scene); 
void readme();


/* @function main */
int main( int argc, char** argv )
{
	// if argument is missing
	if( argc != 3 )
	{ readme(); return -1; }

	// load classifier
	if( !face_cascade.load( face_cascade_name ) )
	{ printf("--(!)Error loading\n"); return -1; };

	// Load image
	Mat img_1 = imread( argv[1], IMREAD_GRAYSCALE );
	Mat img_2 = imread( argv[2], IMREAD_GRAYSCALE );

	vector<Rect> faces_1 = findFaces( img_1 );
 	cout << "image_1 contain:" + to_string(faces_1.size()) << endl;
 	// write img for debug
 	// drawRectagleOnFace(img_1, faces_1);
	vector<Rect> faces_2 = findFaces( img_2 );
 	cout << "image_2 contain:" + to_string(faces_2.size()) << endl;
	drawRectagleOnFace(img_2, faces_2);

	// Loop over the faces from image 1
	for(int i = 0; i < faces_1.size(); i++ ){
		// Crop the only the face section
		Mat face_1 = img_1(faces_1[i]);

		// Compare the face from image 1 to all faces in image 2
		for(int j = 0; j < faces_2.size(); j++ ){
			String pairName = to_string(i) + "_" + to_string(j);
			cout << "check face:" + pairName << endl;
			Mat face_2 = img_2(faces_2[j]);
			/*--- check homography in the second faces ---*/
			try {
				auto [H, img_matches] = checkHomo(face_1, face_2);
				//-- Get the corners from the image_1 ( the object to be "detected" )
				vector<Point2f> obj_corners(4);
				obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( face_1.cols, 0 );
				obj_corners[2] = cvPoint( face_1.cols, face_1.rows ); obj_corners[3] = cvPoint( 0, face_1.rows );
				vector<Point2f> scene_corners(4);
				perspectiveTransform( obj_corners, scene_corners, H);
				//-- Draw lines between the corners (the mapped object in the scene - image_2 )
				line( img_matches, scene_corners[0] + Point2f( face_1.cols, 0), scene_corners[1] + Point2f( face_1.cols, 0), Scalar(0, 255, 0), 4 );
				line( img_matches, scene_corners[1] + Point2f( face_1.cols, 0), scene_corners[2] + Point2f( face_1.cols, 0), Scalar( 0, 255, 0), 4 );
				line( img_matches, scene_corners[2] + Point2f( face_1.cols, 0), scene_corners[3] + Point2f( face_1.cols, 0), Scalar( 0, 255, 0), 4 );
				line( img_matches, scene_corners[3] + Point2f( face_1.cols, 0), scene_corners[0] + Point2f( face_1.cols, 0), Scalar( 0, 255, 0), 4 );
				//-- Show detected matches
				imwrite( "result"+pairName+".jpg", img_matches );
			} catch (const std::exception& e) {
				cout << "Bad Faces" << endl;
			}
		}
	}
	return 0;
}

/** @function drawRectagleOnFace */
void drawRectagleOnFace(Mat img, vector<Rect> faces){
	for(int i = 0; i < faces.size(); i++){
	 	rectangle(img, faces[i], Scalar( 255, 0, 0 ));
	}
 	imwrite("debug.jpg", img);
}

/** @function findFaces */
vector<Rect> findFaces( Mat frame ){
	vector<Rect> faces;
	//-- Detect faces
	face_cascade.detectMultiScale( frame, faces, 1.1, 2, 0, Size(30, 30) );
	return faces;
}

/** @function checkHomo */
tuple<Mat, Mat> checkHomo(Mat img_object, Mat img_scene){
	//-- Step 1: Detect the keypoints and extract descriptors using SURF
	int minHessian = 400;
	Ptr<SURF> detector = SURF::create( minHessian );
	vector<KeyPoint> keypoints_object, keypoints_scene;
	Mat descriptors_object, descriptors_scene;
	detector->detectAndCompute( img_object, Mat(), keypoints_object, descriptors_object );
	detector->detectAndCompute( img_scene, Mat(), keypoints_scene, descriptors_scene );
	//-- Step 2: Matching descriptor vectors using FLANN matcher
	FlannBasedMatcher matcher;
	vector< DMatch > matches;
	matcher.match( descriptors_object, descriptors_scene, matches );
	double max_dist = 0; double min_dist = 100;
	//-- Quick calculation of max and min distances between keypoints
	for( int i = 0; i < descriptors_object.rows; i++ ){ 
		double dist = matches[i].distance;
		if( dist < min_dist ) min_dist = dist;
		if( dist > max_dist ) max_dist = dist;
	}
		printf("-- Max dist : %f \n", max_dist );
		printf("-- Min dist : %f \n", min_dist );
	//-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
	vector< DMatch > good_matches;
	for( int i = 0; i < descriptors_object.rows; i++ ){ 
		if( matches[i].distance <= 3*min_dist ){ 
			good_matches.push_back( matches[i]); 
		}
	}
	Mat img_matches;
	drawMatches( img_object, keypoints_object, img_scene, keypoints_scene,
	           good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
	           vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
	//-- Localize the object
	vector<Point2f> obj;
	vector<Point2f> scene;
	for( size_t i = 0; i < good_matches.size(); i++ ){
		//-- Get the keypoints from the good matches
		obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
		scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
	}
    if (good_matches.size() < 4) {
        throw runtime_error("Insufficient matches to compute homography.");
    }
	Mat H = findHomography( obj, scene, RANSAC );
	return  {H, img_matches};
}


/* @function readme */
void readme()
{ cout << " Usage: ./SURF_descriptor <img_1> <img_2>" << endl; }