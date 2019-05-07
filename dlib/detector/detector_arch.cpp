// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This example program shows how to find frontal human faces in an image and
    estimate their pose.  The pose takes the form of 68 landmarks.  These are
    points on the face such as the corners of the mouth, along the eyebrows, on
    the eyes, and so forth.  
    


    The face detector we use is made using the classic Histogram of Oriented
    Gradients (HOG) feature combined with a linear classifier, an image pyramid,
    and sliding window detection scheme.  The pose estimator was created by
    using dlib's implementation of the paper:
       One Millisecond Face Alignment with an Ensemble of Regression Trees by
       Vahid Kazemi and Josephine Sullivan, CVPR 2014
    and was trained on the iBUG 300-W face landmark dataset (see
    https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/):  
       C. Sagonas, E. Antonakos, G, Tzimiropoulos, S. Zafeiriou, M. Pantic. 
       300 faces In-the-wild challenge: Database and results. 
       Image and Vision Computing (IMAVIS), Special Issue on Facial Landmark Localisation "In-The-Wild". 2016.
    You can get the trained model file from:
    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2.
    Note that the license for the iBUG 300-W dataset excludes commercial use.
    So you should contact Imperial College London to find out if it's OK for
    you to use this model file in a commercial product.


    Also, note that you can train your own models using dlib's machine learning
    tools.  See train_shape_predictor_ex.cpp to see an example.

    


    Finally, note that the face detector is fastest when compiled with at least
    SSE2 instructions enabled.  So if you are using a PC with an Intel or AMD
    chip then you should enable at least SSE2 instructions.  If you are using
    cmake to compile this program you can enable them by using one of the
    following commands when you create the build project:
        cmake path_to_dlib_root/examples -DUSE_SSE2_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_SSE4_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_AVX_INSTRUCTIONS=ON
    This will set the appropriate compiler options for GCC, clang, Visual
    Studio, or the Intel compiler.  If you are using another compiler then you
    need to consult your compiler's manual to determine how to enable these
    instructions.  Note that AVX is the fastest but requires a CPU from at least
    2011.  SSE4 is the next fastest and is supported by most current machines.  
*/


#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/opencv/to_open_cv.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
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
// using namespace cv;

/** Global variables */
string descriptor = "./shape_predictor_68_face_landmarks.dat";
bool debug = true;
/** Function Headers */
// void drawRectagleOnFace(Mat img, vector<Rect> faces);
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

    /* Load images */
    // ./detector img_1 img_2
    array2d<rgb_pixel> img_1, img_2;
    load_image(img_1, argv[1]);
    load_image(img_2, argv[2]);
    // Make the image larger so we can detect small faces.
    pyramid_up(img_1);
    pyramid_up(img_2);

    /* Process images 1 */
    /* Detect Face */
    cout << "processing image " << argv[1] << endl;
    std::vector<rectangle> dets_1 = detector(img_1);
    cout << "--Number of faces detected: " << dets_1.size() << endl;
    /* Descriptor */
    std::vector<full_object_detection> faces_1;
    for(int i = 0; i < dets_1.size(); i++){
        full_object_detection face = sp(img_1, dets_1[i]);
        cout << "----number of landmark of face"+to_string(i)+": "<< face.num_parts() << endl;
        faces_1.push_back(face);
    }
    // /* Extract faces */
    dlib::array<array2d<rgb_pixel> > face_chips_1;
    extract_image_chips(img_1, get_face_chip_details(faces_1), face_chips_1);
    if(debug){
        for(int i = 0; i < face_chips_1.size(); i++){
            for(int k = 0; k < faces_1[i].num_parts(); k++){
                draw_solid_circle(img_1, faces_1[i].part(k), 3,rgb_pixel( 255, 255, 0 ) );
            }
        }
        save_jpeg (img_1, "faces_1.jpg" );
    }

    /* Process images 2 */
    /* Detect Face */
    cout << "processing image " << argv[2] << endl;
    std::vector<rectangle> dets_2 = detector(img_2);
    cout << "--Number of faces detected: " << dets_2.size() << endl;
    /* Descriptor */
    std::vector<full_object_detection> faces_2;
    for(int i = 0; i < dets_2.size(); i++){
        full_object_detection face = sp(img_2, dets_2[i]);
        cout << "----number of landmark of face"+to_string(i)+": "<< face.num_parts() << endl;
        faces_2.push_back(face);
    }
    // /* Extract faces */
    dlib::array<array2d<rgb_pixel> > face_chips_2;
    extract_image_chips(img_2, get_face_chip_details(faces_2), face_chips_2);
    if(debug){
        for(int i = 0; i < face_chips_2.size(); i++){
            for(int k = 0; k < faces_2[i].num_parts(); k++){
                draw_solid_circle(img_2, faces_2[i].part(k), 3,rgb_pixel( 255, 255, 0 ) );
            }
        }
        save_jpeg (img_2, "faces_2.jpg" );
    }
    
    /* loop over faces_1 */
    for(int i = 0; i < faces_1.size(); i++){
        cout << "Cal Face1" << endl;
        full_object_detection face_1 = faces_1[i];
        /* Get landmark_1 */
        std::vector<cv::Point2f> landmark_1(face_1.num_parts());
        for(int k = 0; k < face_1.num_parts(); k++){
            // cout << "get landmark:" + to_string(k) << endl;
            landmark_1[k] = cv::Point2f(face_1.part(k).x(), face_1.part(k).y());
        }
        /* loop over faces_2 */
        for(int j = 0; j < faces_2.size(); j++){
            cv::String pairName = to_string(i) + "_" + to_string(j);
            // cout << "check face:" + pairName << endl;
            cout << "Compare Face:" + to_string(i) + " " + to_string(j) << endl;
            full_object_detection face_2 = faces_2[j];
            /* Get landmark_2 */
            std::vector<cv::Point2f> landmark_2(face_2.num_parts());
            for(int k = 0; k < face_2.num_parts(); k++){
                landmark_2[k] = cv::Point2f(face_2.part(k).x(), face_2.part(k).y());
            }
            try {
                cv::Mat H = cv::findHomography( landmark_1, landmark_2 );
                cout << "H = "<< endl << " "  << H << endl << endl;
                cv::Mat mat_1 = toMat(img_1);
                cv::Mat mat_2 = toMat(img_2);
                std::vector<cv::Point2f> obj_corners(4);
                obj_corners[0] = cvPoint(0,0); 
                obj_corners[1] = cvPoint( mat_1.cols, 0 );
                obj_corners[2] = cvPoint( mat_1.cols, mat_1.rows ); 
                obj_corners[3] = cvPoint( 0, mat_1.rows );
                std::vector<cv::Point2f> scene_corners(4);
                perspectiveTransform( obj_corners, scene_corners, H);
                cv::line( mat_2, scene_corners[0], scene_corners[1], cv::Scalar(0, 255, 0), 4 );
                cv::line( mat_2, scene_corners[1], scene_corners[2], cv::Scalar( 0, 255, 0), 4 );
                cv::line( mat_2, scene_corners[2], scene_corners[3], cv::Scalar( 0, 255, 0), 4 );
                cv::line( mat_2, scene_corners[3], scene_corners[0], cv::Scalar( 0, 255, 0), 4 );
                cv::imwrite( "result"+pairName+".jpg", mat_2 );
            } catch (const std::exception& e) {
                cout << "Bad Faces" << endl;
            }
        }
    }



    // /*  */
    // for (int i = 0; i < imgs.size(); i++){
    //     array2d<rgb_pixel> img = imgs[i];
    //     /* Loop over faces to get descriptor */
    //     vector<full_object_detection> shapes;
    //     for(int j = 0; j < faces.size(); j++){
    //         full_object_detection shape = sp(img, dets[j]);
    //     }
    // }

    //         // Now tell the face detector to give us a list of bounding boxes
    //         // around all the faces in the image.

    //         // Now we will go ask the shape_predictor to tell us the pose of
    //         // each face we detected.
    //         std::vector<full_object_detection> shapes;
    //         for (unsigned long j = 0; j < dets.size(); ++j)
    //         {
    //             cout << "number of parts: "<< shape.num_parts() << endl;
    //             cout << "pixel position of first part:  " << shape.part(0) << endl;
    //             cout << "pixel position of second part: " << shape.part(1) << endl;
    //             // You get the idea, you can get all the face part locations if
    //             // you want them.  Here we just store them in shapes so we can
    //             // put them on the screen.
    //             shapes.push_back(shape);
    //             draw_rectangle(img, dets[j], rgb_pixel( 255, 0, 0 ) );
    //             // draw a point detected on the face
    //             for(int k = 0; k < shape.num_parts(); k++){
    //                 draw_solid_circle(img, shape.part(k), 10,rgb_pixel( 255, 255, 0 ) );
    //             }
    //         }

    //         save_jpeg (img, "debug.jpg" );

    //         // We can also extract copies of each face that are cropped, rotated upright,
    //         // and scaled to a standard size as shown here:

    //         // win_faces.set_image(tile_images(face_chips));

    //         cout << "Hit enter to process the next image..." << endl;
    //         cin.get();
    //     }
    // }
    // catch (exception& e)
    // {
    //     cout << "\nexception thrown!" << endl;
    //     cout << e.what() << endl;
    // }
}

/** @function drawRectagleOnFace */
// void drawRectagleOnFace(Mat img, vector<Rect> faces){
//     for(int i = 0; i < faces.size(); i++){
//         rectangle(img, faces[i], Scalar( 255, 0, 0 ));
//     }
//     imwrite("debug.jpg", img);
// }
// ----------------------------------------------------------------------------------------
/** @function checkHomo */
// Mat checkHomo(Mat img_object, Mat img_scene){

//     Mat H = findHomography( obj, scene, RANSAC );
//     return  H;
// }

/* @function readme */
void readme()
{ cout << " Usage: ./detector shape_predictor_68_face_landmarks.dat ../me/attack5.jpg " << endl; }