import cv2
import sys
import os 
import dlib

def play(CAPTURE):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    # start stream
    CAPTURE.read()
    # win = dlib.image_window()``
    while True:
        status, image = CAPTURE.read()
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        dets = detector(image)
        for det in dets:
            # Bound for face
            cv2.rectangle(image,(det.left(), det.top()), (det.right(), det.bottom()), (0,255,0), 3)
            # Get the landmarks/parts for the face in box d.
            landmarks = predictor(image, det)
            for part in landmarks.parts():
                x = part.x
                y = part.y
                cv2.circle(image, (x,y) , radius=3, color=(0, 0, 255), thickness=-1 )
        cv2.imshow('my webcam', image)
        if cv2.waitKey(10) == 13:
            # Press Enter to exit
            break
    cv2.destroyAllWindows()


def main():
    # Read from video of webcam
    # cv2.VideoCapture(0) #Means first camera or webcam.
    # cv2.VideoCapture(1) #Means second camera or webcam.
    # cv2.VideoCapture(“file name.mp4”) #Means video file
    CAPTURE = None
    if(len(sys.argv) == 1):
        # Default to webcam
        CAPTURE = cv2.VideoCapture(0)
    elif(len(sys.argv) == 2):
        if(os.path.exists(sys.argv[1]) == False):
            raise ValueError(f"{sys.argv[1]} is not exist. This must be full path to a video.")
        # The argument must be path to video
        CAPTURE = cv2.VideoCapture(sys.argv[1])
    else:
        raise NotImplementedError(f"We did not expect argument more than 1")
    play(CAPTURE)



if __name__ == "__main__":
    main()