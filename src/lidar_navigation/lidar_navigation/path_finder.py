import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PointStamped, PoseStamped
from std_msgs.msg import Header
import numpy as np
from utilities.path_finding import astar
import settings as cfg
from tf2_msgs.msg import TFMessage

class PathFinderNode(Node):
    
    def __init__(self):
        Node.__init__(self, 'path_finder_node')

        self.free_space_sub = self.create_subscription(
            OccupancyGrid, 'free_space_grid_raw', self.free_space_callback, 10
        )

        self.robot_pose_sub = self.create_subscription(
            TFMessage, '/tf', self.robot_pose_callback, 10
        )
        self.goal_sub = self.create_subscription(
            PointStamped, '/clicked_point', self.goal_callback, 10
        )

        # Publishers
        self.path_pub = self.create_publisher(Path, 'calculated_path', 10)
        self.simplified_path_pub = self.create_publisher(Path, 'simplified_path', 10)

        self.map_width = cfg.map_width
        self.map_height = cfg.map_height
        self.CELL_SIZE = cfg.CELL_SIZE

        self.free_space_grid = None
        self.goal_location = None
        self.robot_location = (0, 0)

    def free_space_callback(self, msg):
        self.free_space_grid = self.grid_to_numpy(msg)
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        self.get_logger().info('Updated free space grid.')
        self.calculate_path()

    def robot_pose_callback(self, msg):
        # Extract the robot's position in the map frame
        for transform in msg.transforms:
            if transform.child_frame_id == 'base_link':
                #store robot location in cm
                self.robot_location = (transform.transform.translation.x * 100,
                                        transform.transform.translation.y * 100)
                self.get_logger().info(f'Updated robot location: {self.robot_location}')
                self.calculate_path()
                break

    def goal_callback(self, msg):

        #store goal location in cm
        self.goal_location = (msg.point.x * 100, 
                            msg.point.y * 100)
        self.get_logger().info(f'Received goal location: {self.goal_location}')
        self.calculate_path()

    def grid_to_numpy(self, grid_msg):
        data = np.array(grid_msg.data, dtype=np.int8).reshape(grid_msg.info.height, grid_msg.info.width)
        return data

    def calculate_path(self):
        
        try:
            if self.free_space_grid is None or self.goal_location is None:
                self.get_logger().info('Waiting for free space grid and goal location...')
                return

            robot_cell = (
                int(round(self.robot_location[0] / self.CELL_SIZE) + self.map_width / 2),
                int(round(self.robot_location[1] / self.CELL_SIZE) + self.map_height / 2)
            )
            goal_cell = (
                int(round(self.goal_location[0] / self.CELL_SIZE) + self.map_width / 2),
                int(round(self.goal_location[1] / self.CELL_SIZE) + self.map_height / 2)
            )

            path, simplified_path = astar(
                robot_cell, goal_cell, self.free_space_grid, self.map_height, self.map_width
            )

            if path is not None:
                self.get_logger().info('Path Calculated Successfully')
                path_msg = self.array_to_path(path)
                self.path_pub.publish(path_msg)

                simplified_path_msg = self.array_to_path(simplified_path)
                self.simplified_path_pub.publish(simplified_path_msg)
            else:
                self.get_logger().info('Path Calculation Failed')

        except Exception as e:
            self.get_logger().error(f'Error calculating path: {e}')

    def array_to_path(self, path):

        # Convert the path to a nav_msgs/Path message
        path_msg = Path()
        path_msg.header = Header()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'  # Adjust the frame_id as needed

        for (x, y) in path:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = ((x - self.map_width / 2) * self.CELL_SIZE + self.CELL_SIZE / 2) / 100
            pose.pose.position.y = ((y - self.map_height / 2) * self.CELL_SIZE + self.CELL_SIZE / 2) / 100
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0  # Assuming no orientation for simplicity
            path_msg.poses.append(pose)

        return path_msg

def main(args=None):
    rclpy.init(args=args)
    path_finder_node = PathFinderNode()
    rclpy.spin(path_finder_node)
    path_finder_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
