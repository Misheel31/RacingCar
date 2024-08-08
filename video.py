import os
import datetime
import cv2
import csv
import random

frameWidth = 640
frameHeight = 480
fps = 20

# Simulated sensor data function
def get_sensor_data():
    steering = random.uniform(-1.0, 1.0)  # Simulate steering angle (-1 to 1)
    speed = random.uniform(20.0, 60.0)    # Simulate speed in km/h (20 to 60)
    throttle = random.uniform(0.0, 1.0)   # Simulate throttle position (0 to 1)
    brake = random.uniform(0.0, 0.5)      # Simulate brake position (0 to 0.5)
    return steering, speed, throttle, brake

# Video capture setup
cap = cv2.VideoCapture(0)  
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 150)

# Directories for saving images
base_save_directory = "D:\\enterprise\\data"
left_save_directory = os.path.join(base_save_directory, "left")
right_save_directory = os.path.join(base_save_directory, "right")
center_save_directory = os.path.join(base_save_directory, "center")

for directory in [left_save_directory, right_save_directory, center_save_directory]:
    if not os.path.exists(directory):
        os.makedirs(directory)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
video_path = os.path.join(base_save_directory, f"captured_video_{timestamp}.avi")
out = cv2.VideoWriter(video_path, fourcc, fps, (frameWidth, frameHeight))

# CSV setup
csv_path = os.path.join(base_save_directory, f"image_log_{timestamp}.csv")
with open(csv_path, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Timestamp', 'Left Image Path', 'Right Image Path', 'Center Image Path', 'Steering', 'Speed', 'Throttle', 'Brake'])

frame_counter = 0

# Keyboard control variables
steering_angle = 0
throttle = 0

while True:
    success, img = cap.read()
    if not success:
        break

    # Display the frame on the screen
    cv2.imshow("Result", img)

    # Get simulated sensor data
    steering, speed, throttle, brake = get_sensor_data()

    # Capture keyboard inputs
    key = cv2.waitKey(1) & 0xFF
    if key == ord('w'):
        throttle += 0.1
    elif key == ord('s'):
        throttle -= 0.1
    elif key == ord('a'):
        steering_angle -= 0.1
    elif key == ord('d'):
        steering_angle += 0.1
    elif key == 27:  # ESC key to exit
        break

    # Override simulated data with keyboard inputs if any
    steering = steering_angle if steering_angle != 0 else steering
    throttle = throttle if throttle != 0 else throttle

    print(f"Steering Angle: {steering}, Throttle: {throttle}")

    out.write(img)

    if frame_counter % fps == 0:  # Save every second
        timestamp_frame = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        left_img_path = os.path.join(left_save_directory, f"left_{timestamp_frame}.png")
        right_img_path = os.path.join(right_save_directory, f"right_{timestamp_frame}.png")
        center_img_path = os.path.join(center_save_directory, f"center_{timestamp_frame}.png")

        # Save images
        cv2.imwrite(left_img_path, img)
        cv2.imwrite(right_img_path, img)
        cv2.imwrite(center_img_path, img)

        # Write to CSV
        with open(csv_path, mode='a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([timestamp_frame, left_img_path, right_img_path, center_img_path, steering, speed, throttle, brake])

    frame_counter += 1

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video saved at {video_path}")
print(f"CSV log saved at {csv_path}")
print(f"Images saved in {left_save_directory}, {right_save_directory}, and {center_save_directory}")
