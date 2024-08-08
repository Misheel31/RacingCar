import cv2
import matplotlib.pyplot as plt
import numpy as np
from movement import movement
import time

controls = movement()
vid = cv2.VideoCapture(0)
distance_from_center = 0
controls.send_angle(int(45),1500)
time.sleep(1)

def blur(frame):
    return cv2.GaussianBlur(frame, (5,5), 20)

def extract_roi(frame):
    height = frame.shape[0]
    width = frame.shape[1]
    polygon = np.array([[(60, height), (0, 350), (width, 350), (width, height)]], dtype=np.int32)

    mask = np.zeros_like(frame)

    if len(frame.shape) == 2: 
        cv2.fillPoly(mask, polygon, 255)
    else: 
        cv2.fillPoly(mask, polygon, (255, 255, 255))

    masked_image = cv2.bitwise_and(frame, mask)
    return masked_image

def filter_white(image):
    hsl = cv2.cvtColor(image,cv2.COLOR_BGR2HLS)
    lower_white = np.array([84,87,83], dtype=np.uint8)
    upper_white = np.array([255,255,255], dtype=np.uint8)
    #lower_white = np.array([0,200,0], dtype=np.uint8)
    #upper_white = np.array([180,255,255], dtype=np.uint8)
    mask = cv2.inRange(hsl, lower_white, upper_white)
    result = cv2.bitwise_and(image, image, mask = mask)
    result = cv2.cvtColor(result, cv2.COLOR_HLS2BGR)
    cv2.imshow("filter_white",result)
    return result

def canny(frame):
    # return cv2.Canny(frame,40, 60)
    binr = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    kernel = np.ones((5, 5), np.uint8) 
    invert = cv2.bitwise_not(binr) 
    erosion = cv2.erode(invert, kernel, iterations=1)
    edges =  cv2.Canny(erosion,20, 20)    
    return edges  

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (3/5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    try:   
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
        left_fit_average = np.average(left_fit, axis=0)
        right_fit_average = np.average(right_fit, axis=0)
        right_line = make_coordinates(image, right_fit_average)
        left_line = make_coordinates(image, left_fit_average)
        return(np.array([left_line, right_line]))
    except Exception as e:
        print(e) 
        
def find_center(frame, lines):
    if lines is None:
       return None
    left = lines[0]
    right = lines[1]
    y1 = left[3]
    y2 = np.shape(frame)[0]
    x1 = int((left[2] + right[2]) / 2)
    x2 = int((left[0] + right[0]) / 2)
    return np.array([x1,y1,x2,y2])
def calculate_steering_angle(center_line, vehicle_center_line,frame):

    try:  
        center_slope = (center_line[1] - center_line[3]) / (center_line[0] - center_line[2])
        vehicle_slope = (vehicle_center_line[1] - vehicle_center_line[3]) / (vehicle_center_line[0] - vehicle_center_line[2])
        print(center_line)
        center_angle = np.arctan(center_slope) * 180 / np.pi
        vehicle_angle = np.arctan(vehicle_slope) * 180 / np.pi
        
        steering_angle = center_angle - vehicle_angle
        
        remapped_angle = 45 - steering_angle
        remapped_angle = max(min(remapped_angle, 90), 1)
        remapped_angle = abs(90-remapped_angle)
        #plt.imshow(frame)
        #plt.plot(np.array([center_line[2],center_line[3]]),np.array([300,np.shape(frame)[0]]),'o')
        #plt.plot(center_line)
        #plt.show()
        return remapped_angle
    except:
        return None

def draw_lines(frame, lines):
    global distance_from_center
    try:
       VEHICLE_CENTER_X = np.shape(frame)[1]//2
       image = np.zeros_like(frame)
       center_line = find_center(frame,lines)
       if lines is not None :
           for line in lines:
               x1, y1, x2, y2  = line.reshape(4)
               cv2.line(image, (x1, y1), (x2, y2),(255,0,0),2)
       if center_line is not None:
           x1, y1, x2, y2  = center_line.reshape(4)   
           cv2.line(image, (x1, y1), (x2, y2),(255,0,0),1)
           #VEHICLE_CENTER change to variable later
           cv2.line(image, (VEHICLE_CENTER_X, y2), (VEHICLE_CENTER_X, y2 - 50), 255,2)
           distance_from_center = int(x2 - x1)
           center_line = find_center(frame_gray, hough_lines)
           vehicle_center_line = np.array([300, np.shape(frame)[0], 300, np.shape(frame)[0] - 50])
           steering_angle = "none"
           if(distance_from_center > -5 and distance_from_center < 5):
              distance_from_center = 0
           if distance_from_center > 0:
              text = "From Center : " + str(steering_angle) + "  turn left :" 
           elif distance_from_center < 0 :
              text = "From Center : " + str(steering_angle) + "  turn right :" 
           else :
              text = "From Center : " + str(steering_angle)
           font = cv2.FONT_HERSHEY_SIMPLEX
           org = (50, 100)
           fontScale = 1
           thickness = 2
           image = cv2.putText(image, "", org, font, fontScale, 255, thickness, cv2.LINE_AA)
       return image
    except Exception as e:
         print("Draw Lines",e)
def calculate_control_value(center_line, vehicle_center, frame):
    radian_to_degree = 57.296
    # center_line_arr = np.array([291, 284, 327, 474])
    # vehicle_center_line_arr = np.array([300, 300, 300, 500])
    angle = 0
    midpoint_of_center = np.array([(center_line[2] + center_line[0]) / 2,
                                   (center_line[3] + center_line[1]) / 2])
    # plt.imshow(frame)
    # plt.plot(vehicle_center[[0,2]],vehicle_center[[1,3]])
    # plt.plot(center_line[[0,2]],center_line[[1,3]])
    # plt.plot(midpoint_of_center,'o')
    # plt.plot([vehicle_center[0], midpoint_of_center[0]], [vehicle_center[1], midpoint_of_center[1]], label='Line to Midpoint')
    # plt.grid(True)
    # plt.title("center_line")
    # plt.xlim(left = 0, right = 500)
    # plt.ylim(bottom = 500, top = 200)
    # plt.show()
    cv2.line(frame, 
             (vehicle_center[0], vehicle_center[1]), 
             (vehicle_center[2], vehicle_center[3]), 
             (155, 45, 45), 1)
    
    cv2.line(frame, 
             (center_line[0], center_line[1]), 
             (center_line[2], center_line[3]), 
             (0, 255, 0), 1)
    cv2.line(frame, 
             (vehicle_center[0], np.shape(frame)[0]), 
             (int(midpoint_of_center[0]), int(midpoint_of_center[1])), 
             (255, 0, 255), 1)
    print(midpoint_of_center)
    print(vehicle_center)
    # cv2.circle(frame, (midpoint_of_center[0], midpoint_of_center[1]), 5, (0, 255, 255), -1)
    angle = np.arctan((vehicle_center[1]-center_line[1])/(vehicle_center[0] - center_line[0]))
    angle = -np.degrees(angle)
    vmin = 1450
    vmax = 1560
    
    # Calculate normalized angle
    normalized_angle = (angle - 45) / (90 - 45)
    
    # Calculate velocity using the formula
    velocity = vmin + (vmax - vmin) * (1 - normalized_angle ** 2)
    print(velocity)
    if(angle < 0):
        angle = angle +180
    angle = int(angle/2)
    if(angle < 15):
        angle = 15
    #if(angle > 80):
      #s  angle = 80
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 100)
    fontScale = 1
    thickness = 2
    frame = cv2.putText(frame, str(angle), org, font, fontScale, 255, thickness, cv2.LINE_AA)
    # remapped_angle = 45 - angle
    # remapped_angle = max(min(remapped_angle, 90), 1)
    # remapped_angle = abs(remapped_angle)
    #print(np.degrees(180-angle))
    #print(int(velocity))
    return frame, angle
while(True):
    ret, frame = vid.read()
    #frame = cv2.resize(frame,((np.shape(frame)[1]//3), (np.shape(frame)[0]//3)))

    if not ret:
        break
    frame_white = frame
    frame_gray = cv2.cvtColor(frame_white, cv2.COLOR_BGR2GRAY)
    frame_blur = blur(frame_gray)
    frame_canny = canny(frame_blur)
    roi = extract_roi(frame_canny)
    hough_lines = cv2.HoughLinesP(roi, 4, np.pi/180, 120, maxLineGap = 40, minLineLength = 20)
    hough_lines = average_slope_intercept(frame_canny, hough_lines)
    hough_image = draw_lines(roi, hough_lines)
    frame_display_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    try:
       image = cv2.addWeighted(hough_image, 0.8, frame_display_gray, 1, 1)
    except Exception as e:
        print("showing image",e)
    #traffic_state = get_traffic_color(frame)
    if True: 
        if hough_lines is not None:
            centre = np.shape(frame)[1]//2
            center_line = find_center(frame_gray, hough_lines)
            vehicle_center_line = np.array([centre, np.shape(frame)[0], centre,int(np.shape(frame)[0]-150)])
            try:
              image,angle = calculate_control_value(center_line, vehicle_center_line,image)
              #cv2.imshow("frame", image)
              #print(steering_angle)
              image = cv2.resize(image,((np.shape(image)[1]//3),(np.shape(image)[0]//3)))
              image_canny = cv2.resize(frame_canny,((np.shape(frame_canny)[1]//4),(np.shape(frame_canny)[0]//4)))
              cv2.imshow("frame", image)
              cv2.imshow("frame_canny", image_canny)
              cv2.imshow("roi",roi)
              controls.send_angle(int(angle),1560)
              #pass
            except Exception as e:
              print("final:",e)
    else:
        pass
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        #controls.send_angle(int(steering_angle),0)
        plt.imshow(roi)
        plt.show()
        break
controls.send_angle(int(45),1500)
vid.release()
cv2.destroyAllWindows()
