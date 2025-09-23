import serial
import time

class Controller:
    """
    A class to control a vibration motor connected to an Arduino.
    """
    def __init__(self, port='COM9', baudrate=9600):
        """
        Initializes the Controller and connects to the specified serial port.
        """
        try:
            self.ser = serial.Serial(port, baudrate, timeout=1)
            print(f"Connected to {port} at {baudrate} baudrate.")
        except serial.SerialException as e:
            print(f"Error connecting to {port}: {e}")
            self.ser = None
        self.time_since_last_vibration = time.time()

    def on(self, duration=None, delay=0):
        """
        Turns the vibration motor on.

        Args:
            duration (float, optional): The time in seconds to vibrate.
                                       If None, vibrates indefinitely.
            delay (float, optional): The time 
        """
        if self.ser:
            if time.time() < self.time_since_last_vibration + delay:
                return
            self.ser.write(b'1')
            if (duration is not None):
                time.sleep(duration)
                self.time_since_last_vibration = time.time()
                self.off()

    def off(self):
        """
        Turns the vibration motor off.
        """
        if self.ser:
            self.ser.write(b'0')

    def close(self):
        """
        Closes the serial connection.
        """
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("Serial connection closed.")

if __name__ == "__main__":
    # This block is for testing the Controller module
    controller = Controller(port='COM9') # Make sure to use your correct COM port

    if controller.ser:
        try:
            while True:
                try:
                    user_input = input("Enter the vibration time in seconds (or 'q' to quit): ")
                    if user_input.lower() == 'q':
                        break
                    
                    vibration_time = float(user_input)
                    if vibration_time > 0:
                        print(f"Vibrating for {vibration_time} seconds...")
                        controller.on(vibration_time)
                        print("Vibration finished.")
                    else:
                        print("Please enter a positive number.")

                except ValueError:
                    print("Invalid input. Please enter a number.")
        
        except KeyboardInterrupt:
            print("\nExiting program.")

        finally:
            controller.close()