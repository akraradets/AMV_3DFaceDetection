#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/opencv.h>
#include <dlib/opencv/to_open_cv.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/image_transforms.h>
#include <dlib/geometry.h>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/opencv.hpp"
#include <opencv2/videoio.hpp>
#include <iostream>
#include <math.h> 


// using namespace dlib;
using namespace cv;
using namespace std;

/** Global variables */
string descriptor = "./shape_predictor_68_face_landmarks.dat";
bool debug = true;
std::vector<cv::Point2f> transform_via_homography(const std::vector<cv::Point2f>& points, const cv::Matx33f& homography);
void calculate(Mat H, std::vector<cv::Point2f> prev, std::vector<cv::Point2f> curr);
dlib::drectangle correctBounding(dlib::drectangle dbound,int i);
Rect expandBounding(Rect r);
string matToString(cv::Mat mat);
cv::Mat homography_warp(cv::Mat src,cv::Mat H);
cv::Mat loadImage(cv::String path);
std::vector<cv::Point2f> getLandmark(dlib::full_object_detection obj);
cv::Mat drawLandmark(cv::Mat mat_1, std::vector<cv::Point2f> landmark_1, cv::Mat mat_2, std::vector<cv::Point2f> landmark_2);
dlib::rectangle openCVRectToDlib(cv::Rect r);
cv::Rect dlibRectangleToOpenCV(dlib::rectangle r);

void readme();
// ----------------------------------------------------------------------------------------

int main(int argc, char** argv){
    /* Load DLIB lib */
    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
    /* Load Face Descriptor */
    // For extract 68 points face landmarks
    dlib::shape_predictor sp;
    dlib::deserialize(descriptor) >> sp;
    std::map<int, dlib::correlation_tracker> trackers;
    std::map<int, dlib::drectangle> boundings;

    cout << "Load video" << endl;
    /* Load a video clip */
    cv::VideoCapture cap(argv[1]);
    // Check if camera opened successfully
    if(!cap.isOpened()){
        cout << "Error opening video stream or file" << endl;
        return -1;
    }
    cout << "Init frame" << endl;
    cv::Mat frame_curr, frame_prev;
    // read first frame
    cap.read(frame_curr);
    /* set video writer*/
    // Define the codec and create VideoWriter object.The output is stored in 'outcpp.avi' file. 
    cv::VideoWriter video("./output/_outcpp.avi",VideoWriter::fourcc('M','J','P','G'),3, Size(frame_curr.cols,frame_curr.rows));

    /* Init tracker for any face in the frame */
    // convert into dlib image
    dlib::cv_image<dlib::bgr_pixel> img_1(frame_curr);
    // detect the face
    std::vector<dlib::rectangle> dets_1 = detector(img_1);
    cout << "--Number of faces detected: " << dets_1.size() << endl;
    // Descriptor 
    // init tracker
    std::map<int, std::vector<cv::Point2f>> landmark_prev;
    for(int i = 0; i < dets_1.size(); i++){
        dlib::full_object_detection face = sp(img_1, dets_1[i]);
        landmark_prev[i] = getLandmark(face);

        dlib::correlation_tracker t;
        t.start_track(img_1, correctBounding(face.get_rect(),i) );
        trackers[i] = t;
        boundings[i] = trackers[i].get_position();
        // boundings[i] = correctBounding(trackers[i].get_position(),i);

    }
    int count = 0;
    while(1){
        if(count % 10 == 0)cout << "process frame: " << count << endl;
        // now the first frame is old
        frame_curr.copyTo(frame_prev);
        // get the new frame. if empty then break
        if (!cap.read(frame_curr)) break;

        /* For each tracker, find faces and its landmarks */
        dlib::cv_image<dlib::bgr_pixel> frame_curr_dlib(frame_curr);
        std::map<int, std::vector<cv::Point2f>> landmark_curr;
        putText(frame_curr, "Frame Num: " + to_string(count) , Point(30,30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0,255,0),3);
        for(int i = 0; i < trackers.size(); i++){
            cout << "--Face id: " << i << endl;
            trackers[i].update(frame_curr_dlib,boundings[i]);
            // boundings[i] = correctBounding(trackers[i].get_position(),i);
            boundings[i] = trackers[i].get_position();
            Rect r = dlibRectangleToOpenCV(boundings[i]);
            // Draw bounding
            rectangle(frame_curr, r, Scalar(0,255,0),3);
            putText(frame_curr, "id: " + to_string(i), Point(r.x,r.y), FONT_HERSHEY_SIMPLEX, 1, Scalar(0,255,0),3);

            // Get only detected face
            cout << r << endl;
            Mat m = frame_curr(r);
            cout << "-- get m" << endl;
            dlib::cv_image<dlib::bgr_pixel> img_face(m);
            std::vector<dlib::rectangle> det = detector(img_face);
            cout << "---- Face Found: " << det.size() << endl;
            if(det.size() == 1){
                dlib::full_object_detection face = sp(img_face, det[0]);
                landmark_curr[i] = getLandmark(face);
                for(int k = 0; k < face.num_parts(); k++){
                    dlib::draw_solid_circle(img_face, face.part(k), 3,dlib::rgb_pixel( 255, 255, 0 ) );
                }
            }
            // dlib::save_jpeg(img_face, "./output/faces_"+to_string(count)+"_"+to_string(i)+".jpg" );


            /* Find Homography */
            // If there is no landmark to compare
            if(landmark_prev.find(i) == landmark_prev.end() || landmark_curr.find(i) == landmark_curr.end()){
                cout << "-------------------no homo--------------------" << endl;
                int offset = 0;
                if(i == 0) offset = 100;
                putText(frame_curr, "no homo" , Point(r.x ,r.y + r.height + 30 + offset), FONT_HERSHEY_SIMPLEX, 1, Scalar(0,255,0),3);
                continue;
            }
            Mat H = findHomography( landmark_prev[i], landmark_curr[i] );
            // Calculate Reproj Error
            // cout << "prev" << landmark_prev[i] << endl;
            // cout << "curr" << landmark_curr[i] << endl;
            // cout <<  "H" << H << endl;
            std::vector<cv::Point2f> landmark_cal = transform_via_homography(landmark_prev[i],H);
            // cout << "calculated" << landmark_cal << endl;
            float error = 0;
            for(int k = 0; k < landmark_cal.size(); k++){
                Point2f p_cur = landmark_curr[i][k];
                Point2f p_cal = landmark_cal[k];
                error += sqrt( pow ((p_cal.x - p_cur.x), 2)  + pow ((p_cal.y - p_cur.y), 2));
            }
            error = error / landmark_cal.size();
            cout << "error_mean" << error << endl;
            putText(frame_curr, "ME: " + to_string(error), Point(r.x + 60,r.y), FONT_HERSHEY_SIMPLEX, 1, Scalar(0,255,0),3);
            // cout << "diff" << landmark_curr[i] - transform_via_homography(landmark_prev[i],H) << endl;
            // calculate(H,landmark_prev[i],landmark_curr[i]);
            // break;
            // string H_str (H.begin<unsigned char>(), H.end<unsigned char>());
            string H_str = matToString(H);
            // vector<string> H_s;
            size_t pos = 0;
            int row = 1;
            while ((pos = H_str.find("\n")) != std::string::npos) {
                // H_s.push_back(s.substr(0, pos));
                int offset = 0;
                if(i == 0) offset = 100;
                putText(frame_curr, H_str.substr(0, pos), Point(r.x ,r.y + r.height + (row*30) + offset), FONT_HERSHEY_SIMPLEX, 1, Scalar(0,255,0),3);
                H_str.erase(0, pos + 1);
                row++;
            }
            // putText(frame_curr, H_str, Point(r.x ,r.y + r.height + (row*30)), FONT_HERSHEY_SIMPLEX, 1, Scalar(0,255,0),3);
        }

        /* save landmark */
        // landmark_prev = landmark_curr;

        video.write(frame_curr);
        count++;
        // if(count == 2) break;
    }
    // When everything done, release the video capture and write object
    cap.release();
    video.release();
}

void calculate(Mat H, std::vector<cv::Point2f> prev, std::vector<cv::Point2f> curr){
    // std::vector<cv::Point2f> transform_via_homography(prev,H);
    // std::vector<cv::Point2f>
    //     homogeneous.resize(non_homogeneous.size());
    // for (size_t i = 0; i < non_homogeneous.size(); i++) {
    //     homogeneous[i].x = non_homogeneous[i].x;
    //     homogeneous[i].y = non_homogeneous[i].y;
    //     homogeneous[i].z = 1.0;
    // }
    //     for (size_t i = 0; i < ph.size(); i++) {
    //     ph[i] = homography*ph[i];
    // }
    // for(int i = 0; i < landmark_prev)
}

dlib::drectangle correctBounding(dlib::drectangle dbound,int i){
    Rect bound = dlibRectangleToOpenCV(dbound);
    bound = expandBounding(bound);
    return openCVRectToDlib(bound);
}


Rect expandBounding(Rect r){
    int expand = ((r.height + r.width)/2)*0.3 ;
    r += cv::Size(expand, expand);
    r -= cv::Point(expand/2, expand/2 );
    return r;
}

string matToString(cv::Mat mat){
    std::stringstream buffer;
    buffer << mat << std::endl;
    cout << buffer.str() << endl;
    return buffer.str();
}

cv::Mat loadImage(cv::String path){
	cv::Mat mat = cv::imread( path );
	// resize(mat,mat,cv::Size(1920,1080));//resize image
	return mat;
}

// https://stackoverflow.com/questions/34871740/convert-opencvs-rect-to-dlibs-rectangle
dlib::rectangle openCVRectToDlib(cv::Rect r){
  return dlib::rectangle((long)r.tl().x, (long)r.tl().y, (long)r.br().x - 1, (long)r.br().y - 1);
}

cv::Rect dlibRectangleToOpenCV(dlib::rectangle r){
  return cv::Rect(cv::Point2i(r.left(), r.top()), cv::Point2i(r.right() + 1, r.bottom() + 1));
}

std::vector<cv::Point2f> getLandmark(dlib::full_object_detection obj){
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