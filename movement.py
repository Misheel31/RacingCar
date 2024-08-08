import serial
import time
import random

class movement:
      def __init__(self):
          self.ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=.5)
          self.ser.write("0,1500\n".encode())


      def send_angle(self,angle,acc):
          if 0 <= angle <= 90:
              self.ser.write(f"{angle},{acc}\n".encode())
              response = self.ser.readline().decode().strip()
              print(f"ESP response: {response}")
          else:
              print("Angle must be between 0 and 180")

if __name__ == "__main__":
    movements = movement()
    while True:
        try:
            while True:
                angle = int(input("Enter angle: "))
                acc = int(input("Enter acceleration: "))
                movements.send_angle(angle,acc)
                time.sleep(1)
        except ValueError:
            print("Please enter a valid integer")