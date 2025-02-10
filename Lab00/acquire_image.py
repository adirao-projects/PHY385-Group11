import numpy as np
import matplotlib.pyplot as plt
from vmbpy import *
import cv2
import sys

with Vimba.get_instance() as vimba:
     cams = vimba.get_all_cameras()
     with cams[0] as cam:
         # Aquire single frame synchronously
         for frame in cam.get_frame_generator(limit=10):
             if frame.get_status() == FrameStatus.Complete:
                 print('{} acquired {}'.format(cam, frame))
                 cv2.imwrite("image.jpg",frame.as_opencv_image())