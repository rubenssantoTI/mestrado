import cv2
import sys

import argparse
import os

import posenet


parser = argparse.ArgumentParser()
parser.add_argument('--video', type=str, default='video.mp4')
parser.add_argument('--image_dir', type=str, default='./images')
parser.add_argument('--output_dir', type=str, default='./output')
args = parser.parse_args()

vidcap = cv2.VideoCapture(args.video)


def getFrame(sec, count):
    res = sec * 1000
    vidcap.set(cv2.CAP_PROP_POS_MSEC, res)
    hasFrames, image = vidcap.read()
    print("number ", count)
    if hasFrames:
        cv2.imwrite("./images/image"+str(count)+".jpg", image)     # save frame as JPG file
   
    return hasFrames


def main():
  count = 1
  sec = 0
  frameRate = 0.03 

  success = getFrame(sec, count)
  while success:
    count = count + 1
    sec = sec + frameRate
    sec = round(sec, 2)
    success = getFrame(sec, count)


if __name__ == "__main__":
    main()