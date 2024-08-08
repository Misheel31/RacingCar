import cv2
import numpy as np
import matplotlib.pyplot as plt


# Canny Edge Detection
def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

# Region of Interest
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

# Draw Lines
def draw_lines(img, lines, color=[0, 255, 0], thickness=2):  
    if lines is None:
        return
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

# Hough Transform
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines

# Weighted Image
def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    return cv2.addWeighted(initial_img, α, img, β, λ)

# Lane Line Filtering and Polynomial Fitting
def filter_and_fit_lines(lines, img_shape):
    left_fit = []
    right_fit = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:  # Avoid division by zero
                continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 0.5:  # Filter out near-horizontal lines
                continue
            intercept = y1 - (slope * x1)
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0) if left_fit else None
    right_fit_average = np.average(right_fit, axis=0) if right_fit else None

    left_line = create_line(img_shape, left_fit_average) if left_fit_average is not None else None
    right_line = create_line(img_shape, right_fit_average) if right_fit_average is not None else None

    return np.array([left_line, right_line]) if left_line is not None and right_line is not None else None

def create_line(img_shape, line_params):
    if line_params is None:
        return None
    slope, intercept = line_params
    if slope == 0:  # Avoid division by zero
        return None
    y1 = img_shape[0]
    y2 = max(100, int(y1 * 0.6))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return [x1, y1, x2, y2]

# Lane Detection Pipeline
def lane_detection_pipeline(image):
    edges = canny(image, 50, 150)

    imshape = image.shape
    vertices = np.array([[(0, imshape[0]), (imshape[1] // 2 - 100, 100),
                          (imshape[1] // 2 + 100, 100), (imshape[1], imshape[0])]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)
    cv2.imshow("roi",masked_edges)

    # Plotting the ROI for visualization and adjustment
    plt.figure()
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.plot([vertices[0][0][0], vertices[0][1][0]], [vertices[0][0][1], vertices[0][1][1]], color='yellow', linestyle='-', linewidth=2)
    plt.plot([vertices[0][1][0], vertices[0][2][0]], [vertices[0][1][1], vertices[0][2][1]], color='yellow', linestyle='-', linewidth=2)
    plt.plot([vertices[0][2][0], vertices[0][3][0]], [vertices[0][2][1], vertices[0][3][1]], color='yellow', linestyle='-', linewidth=2)
    plt.plot([vertices[0][3][0], vertices[0][0][0]], [vertices[0][3][1], vertices[0][0][1]], color='yellow', linestyle='-', linewidth=2)
    plt.show()

    lines = hough_lines(masked_edges, 1, np.pi / 180, 15, 40, 20)

    line_image = np.zeros_like(image)

    if lines is not None:
        filtered_lines = filter_and_fit_lines(lines, imshape)
        if filtered_lines is not None:
            draw_lines(line_image, [filtered_lines])

    result = weighted_img(line_image, image, 0.8, 1, 0)
    return result

# Placeholder functions for vehicle control
def steer_vehicle(direction):
    print(f"Steering {direction}")

def control_speed(speed):
    print(f"Setting speed to {speed}")

# Adjust this function based on lane position
def lane_following_control(left_line, right_line, frame_width):
    if left_line is not None and right_line is not None:
        # Calculate the center of the lane
        left_x2 = left_line[2]
        right_x2 = right_line[2]
        lane_center = (left_x2 + right_x2) // 2
        frame_center = frame_width // 2

        # Determine the steering direction
        if lane_center < frame_center - 20:
            steer_vehicle("left")
        elif lane_center > frame_center + 20:
            steer_vehicle("right")
        else:
            steer_vehicle("straight")
        control_speed(50) 
    else:
        steer_vehicle("straight")
        control_speed(0)  # Stop if no lane lines are detected

# Main Function
def main():
    cap = cv2.VideoCapture(1)  
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Resize frame for better performance
        frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))

        lane_detected_frame = lane_detection_pipeline(frame)
        # Debugging outputs to visualize intermediate results
        edges = canny(frame, 50, 150)

        cv2.imshow('Original', frame)
        cv2.imshow('Edges', edges)
        cv2.imshow('Lane Detection', lane_detected_frame)

        # Perform lane following control
        imshape = frame.shape
        lines = hough_lines(edges, 1, np.pi / 180, 15, 40, 20)
        if lines is not None:
            filtered_lines = filter_and_fit_lines(lines, imshape)
            if filtered_lines is not None:
                left_line, right_line = filtered_lines
                lane_following_control(left_line, right_line, frame.shape[1])

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
