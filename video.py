#!/usr/bin/python3
import cv2
from darkflow.net.build import TFNet
import numpy as np
import  time

option = {
        'model' : '/home/ezioauditore/Desktop/projects/darkflow/cfg/yolov3-tiny.cfg',
        'load'   : '/home/ezioauditore/Downloads/yolov3-tiny.weights',
        'threshold' : 0.3,
        'gpu': 1.0
}

tfnet=TFNet(option)

capture=cv2.VideoCapture('v.mp4')
colors=[tuple(255*np.random.rand(3)) for i in range(5)]
'''
for color in colors:
    print(color)
'''

while capture.isOpened():
    stime=time.time()
    status,frame=capture.read()
    results = tfnet.return_predict(frame)
    if status:
        for color,result in zip(colors,results):
            # choosing top left for dog
            tl_dog=(result['topleft']['x'],result['topleft']['y'])
            # choosing bottomrigh for dog
            br_dog=(result['bottomright']['x'],result['bottomright']['y'])
            # choosing label for dog
            lb_dog=result['label']
            frame=cv2.rectangle(frame,tl_dog,br_dog,color,4)
            frame=cv2.putText(frame,lb_dog,tl_dog,cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
        cv2.imshow('frr',frame)
        print('FPS {:.1f}'.format(1/(time.time() - stime )))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else :
        capture.release()
        cv2.destroyAllWindows()
    break
