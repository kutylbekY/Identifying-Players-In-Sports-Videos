import cv2
import os

# Input video file path
video_path = 'C:/Users/User/Desktop/UoM 3rd Year Project/Code/video.mov'

# Output folder to save the modified video
output_folder = 'C:/Users/User/Desktop/UoM 3rd Year Project/Code/output/'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Initialize the video capture
capture = cv2.VideoCapture(video_path)

# Get the video's frame width, height, and frames per second (fps)
frame_width = int(capture.get(3))
frame_height = int(capture.get(4))
fps = capture.get(5)

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video_path = os.path.join(output_folder, 'output_video.avi')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

frame_nr = 0

while True:
    success, frame = capture.read()

    if success:
        # Modify the frame (e.g., draw a rectangle)
        cv2.rectangle(frame, (100, 100), (frame_width - 100, frame_height - 100), (0, 255, 0), 2)

        # Write the modified frame to the output video
        out.write(frame)

        frame_nr += 1
    else:
        break

capture.release()
out.release()

print(f'Modified video saved to {output_video_path}')
