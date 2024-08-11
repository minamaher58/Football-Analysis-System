from ultralytics import YOLO

import utils

model = YOLO("Models/best.pt")

# results = model.predict('Input_videos/08fd33_4.mp4',save=True)

# model = YOLO('yolov8l.yaml').load('Models/yolov8/yolov8l.pt')

model.predict('Input_videos/08fd33_4.mp4',save=True)

# trying to predict the frames
# frames = utils.read_video('Input_videos/08fd33_4.mp4')
# batch_size = 20
# results = []
# for i in range(0,len(frames),batch_size):
#     batch = frames[i:i+batch_size]
#     results_batch = model.predict(batch, conf=0.8, save=True)
#     results += results_batch


# print(results[0])
# print('=====================================')
# for box in results[0].boxes:
#     print(box)