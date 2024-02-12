#   Abdur Raheem    20i-0917

#from ultralytics import YOLO

#------------------------------------------------------------------------------
# This is the code to train the model i.e(yolov8n.pt) and save it in the same directory
#------------------------------------------------------------------------------

# # Create a new YOLO model from scratch
# model = YOLO('yolov8n.yaml')

# # Load a pretrained YOLO model (recommended for training)
# model = YOLO('yolov8n.pt')

# # Train the model using the 'coco128.yaml' dataset for 3 epochs
# results = model.train(data='coco128.yaml', epochs=3)

# # Evaluate the model's performance on the validation set
# results = model.val()

# # Perform object detection on an image using the model
# results = model('https://ultralytics.com/images/bus.jpg')

# # Export the model to ONNX format
# success = model.export(format='onnx')

#from ultralytics import YOLO


#------------------------------------------------------------------------------
# Run the pretrained model on our traffic video
#------------------------------------------------------------------------------
import cv2
from ultralytics import YOLO
from supervision.video.dataclasses import VideoInfo
from supervision.video.sink import VideoSink
from supervision.video.source import get_video_frames_generator

model = YOLO('yolov8n.pt')  # pass any model type
model.fuse()
detections = model(source="traffic_srinagarhighway_Islamabad_Police.mp4", show=True, conf=0.4, save=True)

video_info = VideoInfo.from_video_path(source="traffic_srinagarhighway_Islamabad_Police.mp4")
generator = get_video_frames_generator(source="traffic_srinagarhighway_Islamabad_Police.mp4")
