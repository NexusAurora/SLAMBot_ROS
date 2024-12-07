import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from nav_msgs.msg import Path
from tf2_msgs.msg import TFMessage
import json
import time
import math
from scipy.spatial.transform import Rotation as R
from utilities.movement import generate_movement_commands
from std_msgs.msg import Float64MultiArray
import settings as cfg

class NavigationNode(Node):

    def __init__(self):
        Node.__init__(self, 'navigation_node')

        self.odometry_publisher = self.create_publisher(
            Float64MultiArray, 'odometry', 10
        )

        self.path_sub = self.create_subscription(
            Path, 'simplified_path', self.path_callback, 10
        )

        self.robot_pose_sub = self.create_subscription(
            TFMessage, '/tf', self.robot_pose_callback, 10
        )

        self.command_pub = self.create_publisher(
            String, 'send_to_esp32', 10
        )

        self.simplified_path = []
        self.robot_rotation = 0.0  # Initial rotation w.r.t positive x-axis
        self.robot_location = (0, 0)  # Initial robot location

        # Other Parameters
        self.path_wait_no = cfg.path_wait_no  # Number of times to wait for a path updates before generating commands
        self.path_wait_counter = 0

        self.wait_time  = cfg.wait_time
        self.wait_to_send = cfg.wait_to_send

    def path_callback(self, msg):
        # Convert the path from msg to a list of tuples in cm
        self.simplified_path = [
            (
                pose.pose.position.x * 100,
                pose.pose.position.y * 100
            ) for pose in msg.poses
        ]
        self.get_logger().info(f'Updated simplified path: {self.simplified_path}')

        self.path_wait_counter += 1
        if self.path_wait_counter >= self.path_wait_no:
            self.path_wait_counter = 0
            self.get_logger().info('Generating and sending commands...')
            self.generate_and_send_commands(self.simplified_path)

    def robot_pose_callback(self, msg):
        # Extract the robot's position and orientation
        for transform in msg.transforms:
            if transform.child_frame_id == 'base_link':
                self.robot_location = (
                    transform.transform.translation.x * 100,
                    transform.transform.translation.y * 100
                )
                # Calculate the robot's orientation (rotation) in degrees w.r.t the positive x-axis
                rotation_q = transform.transform.rotation
                self.robot_rotation = self.quaternion_to_euler(rotation_q)
                self.get_logger().info(f'Updated robot location: {self.robot_location}, rotation: {self.robot_rotation}')
                break

    def quaternion_to_euler(self, rotation_q):
        # Convert quaternion to euler angle (yaw) using scipy
        q = [rotation_q.x, rotation_q.y, rotation_q.z, rotation_q.w]
        rotation = R.from_quat(q)
        euler_angles = rotation.as_euler('xyz', degrees=True)
        yaw = euler_angles[2]  # Extract the yaw angle
        return yaw

    def generate_and_send_commands(self, simplified_path=None):

        if simplified_path is None:
            self.get_logger().info('No path available')
            return

        command, approx_pose = generate_movement_commands(simplified_path, self.robot_rotation)
        if command:
            self.get_logger().info(f'Generated Command: {command}')
            self.publish_command(command)
            time.sleep(command['t']/1000)
            self.publish_odometry(approx_pose)
        else:
            self.get_logger().info('No command generated')

    def publish_command(self, command):
        # Convert the command dictionary to a JSON string and publish it
        command_json = json.dumps(command)
        self.command_pub.publish(String(data=command_json))
        self.get_logger().info(f'Published command to ESP32: {command_json}')

    def publish_odometry(self, matrix):
        # Publish the odometry matrix as a Float64MultiArray
        msg = Float64MultiArray()
        msg.data = matrix.flatten().tolist()
        self.odometry_publisher.publish(msg)
        self.get_logger().info(f'Published odometry: {msg.data}')

def main(args=None):
    rclpy.init(args=args)
    navigation_node = NavigationNode()
    rclpy.spin(navigation_node)
    navigation_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
