
import cv2
import os
import sys
import math
import json
import requests
import tflearn
from tflearn.layers.core import *
from tflearn.layers.conv import *
from tflearn.layers.normalization import *
from tflearn.layers.estimator import regression
import paho.mqtt.client as paho
#import writeGif
from PIL import Image
import time

def construct_firenet (x,y, training=False):

    # Build network as per architecture in [Dunnings/Breckon, 2018]

    network = tflearn.input_data(shape=[None, y, x, 3], dtype=tf.float32)

    network = conv_2d(network, 64, 5, strides=4, activation='relu')

    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)

    network = conv_2d(network, 128, 4, activation='relu')

    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)

    network = conv_2d(network, 256, 1, activation='relu')

    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)

    network = fully_connected(network, 4096, activation='tanh')
    if(training):
        network = dropout(network, 0.5)

    network = fully_connected(network, 4096, activation='tanh')
    if(training):
        network = dropout(network, 0.5)

    network = fully_connected(network, 2, activation='softmax')

    # if training then add training hyperparameters

    if(training):
        network = regression(network, optimizer='momentum',
                            loss='categorical_crossentropy',
                            learning_rate=0.001)

    # constuct final model

    model = tflearn.DNN(network, checkpoint_path='firenet',
                        max_checkpoints=1, tensorboard_verbose=2)

    return model

################################################################################

if __name__ == '__main__':

################################################################################

    # construct and display model

    model = construct_firenet (224, 224, training=False)
    print("Constructed FireNet ...")

    model.load(os.path.join("models/FireNet", "firenet"),weights_only=True)
    print("Loaded CNN network weights ...")

################################################################################

    # network input sizes

    rows = 224
    cols = 224

    # display and loop settings

    windowName = "Live Fire Detection - FireNet CNN";
    keepProcessing = True;

################################################################################

    if len(sys.argv) == 2:

        # load video file from first command line argument

        #video = cv2.VideoCapture('rtmp://164.52.200.39:1935/live/test')    
        video = cv2.VideoCapture(sys.argv[1]) #use this to get video file as input
        print("Loaded video ...")

        # create window

        cv2.namedWindow(windowName, cv2.WINDOW_NORMAL);

        # get video properties

        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH));
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_time = round(1000/fps);

        #img_counter = 0
        f = open("log.txt", "r")
        counter = f.read()
        f.close()
        if counter == '':
            counter = -1
        else:
            counter = int(counter) + 1

        flag = 0

        while (keepProcessing):
            # start a timer (to see how long processing and display takes)

            start_t = cv2.getTickCount();

            # get video frame from file, handle end of file

            ret, frame = video.read()
            if not ret:
                print("... end of video file reached");
                break;

            # re-size image to network input size and perform prediction

            small_frame = cv2.resize(frame, (rows, cols), cv2.INTER_AREA)

            # perform prediction on the image frame which is:
            # - an image (tensor) of dimension 224 x 224 x 3
            # - a 3 channel colour image with channel ordering BGR (not RGB)
            # - un-normalised (i.e. pixel range going into network is 0->255)

            output = model.predict([small_frame])

            # label image based on prediction

            if round(output[0][0]) == 1 and flag == 0:
                flag = 1
                #cv2.rectangle(frame, (0,0), (width,height), (0,0,255), 50)
                #cv2.putText(frame,'FIRE',(int(width/16),int(height/4)),
                    #cv2.FONT_HERSHEY_SIMPLEX, 4,(255,255,255),10,cv2.LINE_AA);
                counter += 1
                cv2.imwrite('images/' + 'incident_fire_' + str(counter) + '.jpg', frame)
            #img_counter += 1
            #if img_counter == 150:
                #image_ = cv2.imread('images/' + str(counter) + '.jpg')
                #cv2.imwrite('incident_fire_' + str(counter) + '.jpg', image_12)

                url = 'http://164.52.200.39/models/images/upload.php'
                files = {'uploadedFile': open('images/incident_fire_' + str(counter) + '.jpg', 'rb')}
                #payload = {}
                #headers = {
                #        'Cookie': 'PHPSESSID=ob6en1ji7c6nb015nmfi78tcl1'
                #    }
                response = requests.request("POST", url, 
                                            #headers=headers, data = payload, 
                                            files = files)
                print(files)
                print(response.text.encode('utf8'))

                static_url = 'http://164.52.200.39/models/images/uploaded_files/'
                json_response = {
                        'incident_id': counter,
                        'incident_location': '1056 Zackery Harbor',
                        'incident_date_time': '2020-09-19 13:06:57',
                        'incident_image': static_url + 'incident_fire_' + str(counter) + '.jpg',
                        'incident_type': 'Fire',
                        'alert_dismissed': 0,
                        'authority_info': '912-349-2345',
                        'camera_url': 'camera_url 1',
                        'incident_level': 92
                    }
                #with open('json_response'+str(counter)+'.json', 'w') as write_file:
                #    json.dump(json_response, write_file)

                f = open('log.txt', 'w')
                f.write(str(counter))
                f.close()    
                #series_of_images = []
                #for i in range(img_counter):
                #    series_of_images.append(Image.open('images/' + str(i) + '.jpeg'))

                #writeGif(str(counter) + '.gif', series_of_images, duration=10)
                #break
                broker="broker.mqttdashboard.com"
                port=1883
                def on_publish(client,userdata,result):
                    print("data published \n")
                    pass

                client1= paho.Client("control1")
                client1.on_publish = on_publish
                client1.connect(broker,port)
                ret = client1.publish("test_incident",json.dumps(json_response), 2)
                time.sleep(5)

            else:
                flag = 0
                #cv2.rectangle(frame, (0,0), (width,height), (0,255,0), 50)
                #cv2.putText(frame,'CLEAR',(int(width/16),int(height/4)),
                    #cv2.FONT_HERSHEY_SIMPLEX, 4,(255,255,255),10,cv2.LINE_AA);

            # stop the timer and convert to ms. (to see how long processing and display takes)

            stop_t = ((cv2.getTickCount() - start_t)/cv2.getTickFrequency()) * 1000;

            # image display and key handling

            cv2.imshow(windowName, frame);

            # wait fps time or less depending on processing time taken (e.g. 1000ms / 25 fps = 40 ms)

            key = cv2.waitKey(max(2, frame_time - int(math.ceil(stop_t)))) & 0xFF;
            if (key == ord('x')):
                keepProcessing = False;
            elif (key == ord('f')):
                cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN);

        #broker="broker.mqttdashboard.com"
        #port=1883
        #def on_publish(client,userdata,result):
        #    print("data published \n")
        #    pass

        #client1= paho.Client("control1")
        #client1.on_publish = on_publish
        #client1.connect(broker,port)
        #ret = client1.publish("test_incident",json.dumps(json_response), 0)
    else:
        print("usage: python firenet.py videofile.ext")
################################################################################
