import cv2
import os, sys
from natsort import natsorted, ns

image_folder = sys.argv[1]
#image_folder = "/media/jspinner/shared/Studium/project1/GenerativeJetting/runs/AutoRegGMM_toy_2484/mu_sigma"
video_name = os.path.join(image_folder, "film.mp4")

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
images = natsorted(images)
frame = cv2.imread(os.path.join(image_folder, images[0]))

height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), 5, (width,height))

for image in images:
     video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()
