import serial
import time
from pynput import keyboard
from video import VideoLogger
import cv2
 
class KeyboardController:
    def __init__(self):
        self.ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=.5)  
        time.sleep(2)  # Wait for the serial connection to initialize
        self.ser.write("45,1500\n".encode())
        self.angle = 45
        self.acc = 1525
 
    def send_angle(self):
        if 0 <= self.angle <= 90:
            self.ser.write(f"{self.angle},{self.acc}\n".encode())
            print(f"ESP response: Angle={self.angle}, Acc={self.acc}")
        else:
            print("Angle must be between 0 and 90")
 
    def start_movement(self):
        self.ser.write("45,1525\n".encode())
        self.acc = 1525
        print("Movement started")
 
    def stop_movement(self):
        self.acc = 1500
        self.ser.write("45,1500\n".encode())
        print("Movement stopped")
 
def increase_angle():
    global movements
    movements.angle += 15
    if movements.angle > 90:
        movements.angle = 90
    movements.send_angle()
 
def decrease_angle():
    global movements
    movements.angle -= 15
    if movements.angle < 0:
        movements.angle = 0
    movements.send_angle()
 
def on_press(key):
    try:
        if key == keyboard.Key.up:
            increase_angle()
        elif key == keyboard.Key.down:
            decrease_angle()
        elif key == keyboard.Key.right:
            movements.start_movement()
        elif key == keyboard.Key.left:
            movements.stop_movement()
    except AttributeError:
        pass
 
if __name__ == "__main__":
    global movements
    movements = KeyboardController()
    video_logger = VideoLogger()
 
    with keyboard.Listener(on_press=on_press) as listener:
        print("Press arrow keys to control movement: ")
        print("Up/Down to adjust angle, Right to start movement, Left to stop movement")
 
        while True:
            if not video_logger.log_frame():
                break
 
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
 
        video_logger.release()
        listener.join()