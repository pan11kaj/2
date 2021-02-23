import cv2
import numpy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os,ssl,time

# setting in hhtps context to fetch data from openML:
if (not os.environ.get("PYTHONHTTPSVERIFY") and getattr(ssl,'_create_unverified_contacts',None)):
    ssl._create_default_https_context = ssl._create_unverified_context

# fetch the data
y = pd.read_csv('data.csv')['labels']
X = numpy.load('image.npz')['arr_0']
classes = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
n_classes = len(classes)
#splitting th data scaling: 
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 9,train_size = 7500,test_size = 2500)
# scaling the features:
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

# fitting the train data into the model:
clf = LogisticRegression(solver = 'saga',multi_class = 'multinomial').fit(X_train_scaled,y_train)

# calculating the accuracy of the model
y_pred = clf.predict(X_test_scaled)
acc =  accuracy_score(y_test,y_pred)
print(acc)

# starting the camera
cap = cv2.VideoCapture(0)

while (True):
    # capture fram by fram
    try:
        ret,frame = cap.read()
        #our opreations are come here
        gray = cv2.cvt_Color(frame,cv2.COLOR_BGR2GRAY)
        # drawing a box in the center of the video
        height,widht = gray.shape()
        upper_left = (int(width/2-56),int(height/2-56))
        bottom_right = (int(width/2+56),int(height/2+56))
        cv2.rectangle(gray,upper_left,bottom_right,(0,255,0),2)
        # to only consider Area to identify the digit:
        # roi = region of intrest
        roi = gray[upper_left[1]:bottom_right[1],upper_left[0]:bottom_right[0]]
        # coverting cv2 format to pil format
        im_il = Image.fromArray(roi)
        # convert to gray sclae image - 'L' format means each pixel is :
        #represented by single val 0 to 255
        img_w = im_il.convert('L')
        img_w_resized = ima_w.resize((28,28),Image.ANTIALIAS)
        img_w_resized_inverted = PIL.ImageOps.invert(img_w_resized)
        img_filter = 20
        min_pixel = np.percentile(img_w_resized_inverted,img_filter)
        img_inverted_scaled =  np.clip(img_w_resized_inverted-min_pixel,0,255)
        max_pixel = np.max(img_w_resized_inverted)
        img_inverted_scaled = np.asarray(img_inverted_scaled)/max_pixel
        test_sample = np.array(img_inverted_scaled).reshape(1,7)
        test_predict = clf.predict(test_sample)
        print("predicted scale is: ",test_predict)
        # display the resulting frame
        cv2.im_show('frame',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        pass
    #everything is done release the cpture
    cap.release()
    cv2.destroyAllWindows()
        


