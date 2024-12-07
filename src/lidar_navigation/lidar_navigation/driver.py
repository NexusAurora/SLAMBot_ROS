import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
import socket
import json
import math
import settings as cfg
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utilities.model import Model

class DriverNode(Node):
    
    def __init__(self):
        
        super().__init__('driver_node')
        
        #! Config ----------------------

        self.START_DEGREE = cfg.START_DEGREE
        self.END_DEGREE = cfg.END_DEGREE
        self.NUM_READINGS = cfg.NUM_READINGS

        #! Self port ----------------------

        self.localIP = self.get_local_ip()
        self.localPort = 1234
        self.get_logger().info(f"Own Local IP Address: {self.localIP}")

        #! ESP32 port ----------------------

        self.espIP = None
        self.espPort = 1234 

        #! Socket ----------------------

        self.bufferSize = 2024   
        # Create a datagram socket
        self.serverSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        # Bind to address and ip
        self.serverSocket.bind(("", self.localPort))
        # don't block the socket
        self.serverSocket.setblocking(False)

        self.get_logger().info(f"UDP server up and listening on port {self.localPort}")
        self.publisher_ = self.create_publisher(LaserScan, 'lidar_scan', 10)
        self.timer = self.create_timer(0.1, self.receive_data)  # Call every 0.1 seconds

        #! Model ----------------------

        #load the model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Model().to(self.device)
        self.model.load_state_dict(torch.load('calibration_model.pth', map_location=torch.device('cpu'), weights_only=True))

        #! Sending Data to ESP32 ----------------------

        # Subscriber for sending data back to ESP32
        self.subscription = self.create_subscription(
            String,
            'send_to_esp32',
            self.send_data_callback,
            10
        )
    
    def send_data_callback(self, msg):

        try:
            if self.espIP is not None:
                self.get_logger().info(f"Sending data to ESP32: {msg.data}")
                self.serverSocket.sendto(msg.data.encode('utf-8'), (self.espIP, self.espPort))
            else:
                self.get_logger().warning("ESP32 IP address is not known. Cannot send data.")
        except Exception as e:
            self.get_logger().error(f'Error sending data: {e}')

    def receive_data(self):

        try:
            #* Receiving data coming through UDP
            message, address = self.serverSocket.recvfrom(self.bufferSize)
            message = message.decode('utf-8')  # Decode message to string
            self.get_logger().info(f"Received data from {address}")

            if self.espIP is None:
                self.espIP = address[0]
                self.get_logger().info(f"ESP32 IP Address: {self.espIP}")

            #* Parsing Json
            data = json.loads(message)
            self.get_logger().info(f'Received data: {data}')

            #! Preprocess the data ------------------------------    
            #convert to tensor
            data = torch.tensor(data['f'], dtype=torch.float32).to(self.device)
            data = data.unsqueeze(0)

            #predict
            #* Deep learning model to calibrate, filter and adjust incoming scan values
            data = self.model(data)
            data = data.cpu().detach().numpy().tolist()
            data = data[0]

            #! Populate the LaserScan message ----------------------
            scan_msg = LaserScan()
            scan_msg.header.stamp = self.get_clock().now().to_msg()
            scan_msg.header.frame_id = 'lider_link'  # Assuming base_link frame

            scan_msg.angle_min = math.radians(self.END_DEGREE)  # Assuming 180 degrees end
            scan_msg.angle_max = math.radians(self.START_DEGREE)  # Assuming 0 degrees start
            scan_msg.angle_increment = - math.radians(1) # Example angle increment degree = 1 degree

            scan_msg.time_increment = 0.0  # Set to 0 if not applicable
            scan_msg.scan_time = 0.4  # Example scan time in seconds
            scan_msg.range_min = 30.0 / 1000.0  # Example minimum range (Float)
            scan_msg.range_max = 1000.0  / 1000.0  # Example maximum range (Float)

            scan_msg.ranges = [r / 1000.0 for r in data]  # Convert to meters if necessary
            scan_msg.intensities = []  # Leave empty if not using intensities

            #! Publish the LaserScan message ----------------------
            self.publisher_.publish(scan_msg)

        except BlockingIOError:
            # No data is available, skip
            pass

    def get_local_ip(self):
        try:
            # Create a temporary socket to determine the local IP address
            temp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            temp_socket.connect(("8.8.8.8", 80))  # Connect to an external server
            local_ip = temp_socket.getsockname()[0]
            temp_socket.close()
            return local_ip
        except Exception as e:
            self.get_logger().error(f"Error determining local IP address: {e}")
            return "127.0.0.1"

def main(args=None):
    rclpy.init(args=args)
    driver_node = DriverNode()
    rclpy.spin(driver_node)
    driver_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
