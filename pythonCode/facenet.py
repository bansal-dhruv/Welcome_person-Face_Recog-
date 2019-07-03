import cv2
import numpy as np
import dlib
import glob
from scipy.spatial import distance
from imutils import face_utils
from keras.models import load_model
from fr_utils import *
from inception_blocks_v2 import *
from datetime import date

detector = dlib.get_frontal_face_detector()

FRmodel = load_model('face-rec_Google.h5')
print("Total Params:", FRmodel.count_params())
today = str(date.today())
today=today[-2:]
#predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
thresh = 0.25


def recognize():
    cap = cv2.VideoCapture(0)
    # (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    # (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    while True:
        ret, img = cap.read()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(img_rgb)
        x, y, w, h = 0, 0, 0, 0
        for face in faces:
            (x, y, w, h) = face_utils.rect_to_bb(face)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
            person = img[y:y + h, x:x + w]
            cv2.imshow('Recognizing faces',img)
            name,status=who_is_it(person, database, FRmodel)
            if(status==1):
            	print('Hi '+name)
            elif(status==-1):
            	print('Enter your Name')
            	name=Input()
            	saveName(name,person)
            print(status)
        if cv2.waitKey(1) == ord('q'):
            break




def who_is_it(image, database, model):
    
    encoding = img_to_encoding(image, model)    
    min_dist = 100
    identity = None
    for (name, db_enc) in database.items():
        dist = np.linalg.norm(db_enc - encoding)
        print('distance for %s is %s' %(name, dist))
        if dist < min_dist:
            min_dist = dist
            identity = name

    if identity[-2:]==today:
    	return ('',0)
    elif min_dist > 0.52:
        return (None,-1)
    else:
        changeName(identity)
        return (str(identity),1)
    
def changeName(identity):
    src="/home/dhruv_bansal/Documents/face_recog/images/"
    os.rename(src+identity+'.jpg', src+identity[:-2]+today+'.jpg')
    return 'Done'

def saveName(name,new_img):
    src="/home/dhruv_bansal/Documents/face_recog/images/"
    cv2.imwrite(src+name+today+'.jpg', new_img)
    return 'Done'


def load_database():    
    database = {}
    print("database Creating ...")    
    for file in glob.glob("images/*"):
        identity = os.path.splitext(os.path.basename(file))[0]
        database[identity] = fr_utils.img_path_to_encoding(file, FRmodel)
    return database

database=load_database()
recognize()