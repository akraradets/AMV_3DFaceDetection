import cv2
import sys
import os 
import dlib
import numpy as np

# According to the landmark index
# I pick 0, 8, 16, 30 as points for perspective transformation
PERSPECTIVE_IDX = [0, 8, 16, 19, 24, 30]
MASTER_IMAGE = None
MASTER_FACE = None
MASTER_LANDMARK = None
MASTER_PERSPECTIVE = []
MASTER_DISTANCE = 0

def cal_dist(landmark):
    # It supposes to just a `sqrt(|| x^2 - y^2 ||)`
    pivot = landmark[30]
    return np.linalg.norm(landmark - pivot)

def play(CAPTURE):
    # start stream
    CAPTURE.read()
    while True:
        status, image = CAPTURE.read()
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_image = image.copy()
        faces = detect_face(image)
        for face in faces:
            landmark, perspective = extract_landmark(image, face)
            H,_ = cv2.findHomography(perspective, MASTER_PERSPECTIVE)
            warp_image = cv2.warpPerspective(original_image, H, (original_image.shape[1], original_image.shape[0]))
            landmark = np.concatenate([landmark, np.ones(68).reshape(68,1)], axis=1).reshape(68,3,1)
            warp_landmark = (H@landmark)[:,:2].squeeze()
            for idx, point in enumerate(warp_landmark):
                p = (int(point[0]), int(point[1]))
                cv2.circle(warp_image, p , radius=3, color=(0, 0, 255), thickness=-1 )
                cv2.putText(warp_image, str(idx), p, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0))

            dist = cal_dist(warp_landmark)
            error = abs(MASTER_DISTANCE - dist)
            cv2.putText(warp_image, str(f"Error: {error}"), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0))
            cv2.putText(image, str(f"Error: {error}"), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0))


        cv2.imshow('DEBUG', warp_image)
        cv2.imshow('original', image)
        if cv2.waitKey(10) == 13:
            # Press Enter to exit
            break
    cv2.destroyAllWindows()

def detect_face(image):
    detector = dlib.get_frontal_face_detector()
    faces = detector(image)
    for face in faces:
        # Draw box on face
        cv2.rectangle(image,(face.left(), face.top()), (face.right(), face.bottom()), (0,255,0), 3)
    return faces

def extract_landmark(image, face):
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    landmark = predictor(image, face)
    np_landmark = []
    perspective = []
    # Get the landmarks/parts for the face in box d.
    for idx, part in enumerate(landmark.parts()):
        x = part.x
        y = part.y
        cv2.circle(image, (x,y) , radius=3, color=(0, 0, 255), thickness=-1 )
        cv2.putText(image, str(idx), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0))
        np_landmark.append(np.array([x,y]))
        if(idx in PERSPECTIVE_IDX):
            perspective.append(np.array([x,y]))
    np_landmark = np.vstack(np_landmark)
    perspective = np.vstack(perspective)
    return np_landmark, perspective

def main():
    global MASTER_IMAGE, MASTER_FACE, MASTER_LANDMARK, MASTER_PERSPECTIVE, MASTER_DISTANCE
    # This is the master image to compare with
    MASTER_IMAGE = cv2.imread(cv2.samples.findFile("master.jpg"))
    MASTER_FACE = detect_face(MASTER_IMAGE)
    assert len(MASTER_FACE) == 1
    MASTER_FACE = MASTER_FACE[0]
    MASTER_LANDMARK, MASTER_PERSPECTIVE = extract_landmark(MASTER_IMAGE, MASTER_FACE)
    MASTER_DISTANCE = cal_dist(MASTER_LANDMARK)
    cv2.imshow('MASTER', MASTER_IMAGE)

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