from os import close
import cv2 as cv # main vision library
import numpy as np # To generate random numbers
import dlib # To detect facial features

# Global variables

# initializing dlib classifier
detector_path = 'data/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(detector_path)

EYE_AR_THRESH = 0.3 # Eye aspect ratio threshold


def load_video(path):
    ''' Function to load path'''
    print('Loading Video...')

    video_capture = cv.VideoCapture(path) # Loading video from path
    return video_capture


def getting_random_frames(video_capture):
    ''' Picking random frames from video '''
    print('Picking random frames...')

    images = []
    while True: # In production instead of using infinite loop use 15 sec time
        ret, frame = video_capture.read() # reading all frames

        if not ret:
            print('No frame available') # if no frame is available exit
            return images

        else:
            if np.random.randint(0, 4, 1)[0] == 0: #randomly select frames from video with probability of 0.25
                images.append(cv.cvtColor(frame, cv.COLOR_BGR2GRAY)) # Saving selected frames
        cv.imshow('frame', frame)

        if cv.waitKey(1) & 0xff == ord('q'): # Only for live feed need to change it in production code else infinite loop
            return images


def resizing_image(frames):
    '''Function to resize images to 224 X 224'''
    print('Resizing Image...')

    resized_frame = []
    for frame in frames:
        #width = int(0.60 * frame.shape[1])
        #height = int(0.60 * frame.shape[0])
        resized_frame.append(cv.resize(frame, (250, 250))) # resizing each image to 250 X 250, faster computation
    return resized_frame


def rect_to_bb(rect):
    ''' Converting bounding box of dlib to opencv format'''
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)


def shape_to_np(shape):
    ''' dlib landmark detector returns shape object'''
    coords = np.zeros((12, 2), dtype=int)

    for i in range(0, 12): # Selecting cordinates of only both eyes
        coords[i] = (shape.part(i+36).x, shape.part(i+36).y)
    return coords
        

def detect_face_feature(images):
    '''Function to detect faces in images'''
    print('Detecting faces...')
    cords = [] # To save cordinates of both eyes
    for image in images: # Iterating over all the images
        face_rect = detector(image, 0)
        if len(face_rect): 
            # Determining facial landmark for face, then converting to x, y cordinates
            face_rect = face_rect[0] # Calcualtion on only 1st face found if multiple present
            shape = predictor(image, face_rect)
            shape = shape_to_np(shape)
            cords.append(shape)
    
    return cords


def euclidean_dist(point1, point2):
    '''Function to calculate euclidean distance between two points'''
    return np.linalg.norm(point1 - point2)


def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = euclidean_dist(eye[1], eye[5])
	B = euclidean_dist(eye[2], eye[4])
	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = euclidean_dist(eye[0], eye[3])
	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)
	# return the eye aspect ratio
	return ear


def closed_opened(cords):
    '''Function which takes cordinates and outputs whether person is sleeping or not'''
    print('Detecting eyes...')
    closed, opened = 0, 0
    for cord in cords:
        # Calculating aspect ratio for left eye
        left_eye = cord[0:6] # Cordinates of left eye
        right_eye = cord[6:12] # Cordinates of right eye

        
        left_ear = eye_aspect_ratio(left_eye) # Aspect ratio of left eye
        right_ear = eye_aspect_ratio(right_eye) # Aspect ratio of right eye

        avg_ear = (left_ear + right_ear) / 2.0 # Averaging aspect ratio to judge over all eye state

        if avg_ear < EYE_AR_THRESH: 
            closed += 1
        elif avg_ear >= EYE_AR_THRESH:
            opened += 1

    if closed > opened:
        return False
    else:
        return True


video_path = 'data/video.mp4' # Path of video
video = load_video(2) # Loading video

images = getting_random_frames(video) # Selecting random frames
print(f'Number of frames selected = {len(images)}')

resized_image = resizing_image(images) # resizing images
cordinates = detect_face_feature(resized_image) # Getting cordinates of both eyes

if len(cordinates):
    open_or_not = closed_opened(cordinates) # Getting boolean value whether eye is open or not
    print(f'eyes opened = {open_or_not}')
else:
    print('No face detected') # if no face is detected