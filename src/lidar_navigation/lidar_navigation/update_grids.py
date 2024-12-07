import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header
from geometry_msgs.msg import Pose, Quaternion
from tf2_ros import TransformBroadcaster
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation as R
import numpy as np
import json
import settings as cfg
from utilities.plotter import plot, transform_points_inverse, update_occupancy_grid, update_free_space_grid
from utilities.particle_filter import icp
import cv2
from std_msgs.msg import Float64MultiArray

class UpdateGrids(Node):

    def __init__(self):

        super().__init__('update_grids')

        self.subscription = self.create_subscription(
            LaserScan,
            'lidar_scan',
            self.lidar_callback,
            10)

        self.subscription = self.create_subscription(
            Float64MultiArray,
            'odometry',
            self.update_pose,
            10
        )
        
        self.occupancy_pub = self.create_publisher(OccupancyGrid, 'occupancy_grid', 10)
        self.free_space_pub = self.create_publisher(OccupancyGrid, 'free_space_grid', 10)
        self.occupancy_pub_raw = self.create_publisher(OccupancyGrid, 'occupancy_grid_raw', 10)
        self.free_space_pub_raw = self.create_publisher(OccupancyGrid, 'free_space_grid_raw', 10)

        self.occupancy_grids = [np.zeros((cfg.map_height, cfg.map_width))]
        self.free_space_grids = [np.zeros((cfg.map_height, cfg.map_width))]

        self.previous_coordinates = []
        self.best_coordinates = {
            'coordinates': [],
            'scores': []
        }

        self.occupancy_grids = []
        self.free_space_grids = []

        self.robot_pose = np.array([
            [np.cos(cfg.robot_pose[2]), -np.sin(cfg.robot_pose[2]), cfg.robot_pose[0]],
            [np.sin(cfg.robot_pose[2]),  np.cos(cfg.robot_pose[2]), cfg.robot_pose[1]],
            [0,              0,             1]
        ])

        self.br = TransformBroadcaster(self)
        self.static_broadcaster = StaticTransformBroadcaster(self)

        #Calculating LIDER to BASE_LINK transformation
        self.Lider_y_offset = cfg.body_offset_y / 100 + (2.8 / 100) #in meters

        self.static_broadcaster.sendTransform(self.xyz_to_transform('base_link', 'lider_link', 0.0, self.Lider_y_offset, 0.0, 0.0, 0.0, 0.0))

    def lidar_callback(self, msg):

        # Assuming the msg contains the LiDAR data, process it
        data = msg.ranges  # Extract Array of flotes in meters

        #! Calculations ---------------------------------------------------

        #* Calculate (x, y) coordinates of the LiDAR points ---------------
        coordinates = plot(data, cfg.START_DEGREE, cfg.END_DEGREE, cfg.NUM_READINGS) # -----> [(x1, y1), (x2, y2), ... ]

        #* Mapping & Particle Filtring ------------------------------------
        if(self.previous_coordinates):

            #! ICP ---------------------------------------------------------
            #to numpy array

            #* Considering history to reduce noise and errors
            A = np.concatenate(self.previous_coordinates[-cfg.WINDOW_SIZE-1:])

            '''#if best coordinates are available then add those as well
            if(self.best_coordinates['coordinates']):
                A = np.concatenate([A, self.best_coordinates['coordinates']])
                self.get_logger().info("Best Coordinates Found: "+ str(len(self.best_coordinates['coordinates'])))
            '''

            B = np.array(list(coordinates))
            
            self.get_logger().info("Coordinates Received: "+ str(len(B)))
            
            #check if B is empty
            if B.size == 0:
                self.get_logger().error("No coordinates found [B is empty]")
                return
        
            T_final, distances, iterations = icp(A, B, init_pose=np.linalg.inv(self.robot_pose), max_iterations=cfg.icp_iterations, tolerance=cfg.icp_tolerance)
            self.get_logger().info("ICP Iterations Performed: "+ str(iterations+1))

            #inverse of T
            T_inv = np.linalg.inv(T_final)

            #New coordinates
            #* Fix Robot pose and new coordinates using the transformation matrix
            # Fixing coordinates
            B = transform_points_inverse(T_final, B)
            coordinates = B # -----> [(x1, y1), (x2, y2), ... ]

            #new robot pose
            self.robot_pose = T_inv
        
        #* Update the occupancy grid map ----------------------------------
        occupancy_grid, filtered_coordinates, self.best_coordinates = update_occupancy_grid(coordinates, self.best_coordinates, cfg.map_height, cfg.map_width, cfg.CELL_SIZE, self.occupancy_grids, alpha=0.9)

        #* Storing the calculated occupancy grid in history ---------------
        # Update past occupancy grids
        self.occupancy_grids.append(occupancy_grid)
        if(len(self.occupancy_grids) > cfg.WINDOW_SIZE):
            self.occupancy_grids.pop(0)

            '''for i in range(cfg.erotion_extent):
                #Erode the map
                self.occupancy_grids[i] = (self.occupancy_grids[i]*(1-cfg.erosion_contritubion)) + (cv2.erode(self.occupancy_grids[i], cfg.erosion_kernel, iterations=1)*cfg.erosion_contritubion)
            '''
            for i in range(cfg.probability_filter_extent):
                #Remove low probability cells from older grids
                min_value = np.min(self.occupancy_grids[i])
                max_value = np.max(self.occupancy_grids[i])
                threshold = min_value + (max_value - min_value) * cfg.probability_filter
                self.occupancy_grids[i][self.occupancy_grids[i] < threshold] = 0
            
        #* Storing Each Measurement History -------------------------------
        # Update past coordinates
        # Window size determines how many previous measurements to consider

        self.previous_coordinates.append(np.array(list(filtered_coordinates)))
        if len(self.previous_coordinates) > cfg.WINDOW_SIZE:
            self.previous_coordinates.pop(0)

        #* Update the free space grid map ----------------------------------
        free_space_grid, robot_cell, robot_rotation = update_free_space_grid(self.robot_pose, coordinates, cfg.map_height, cfg.map_width, cfg.CELL_SIZE, self.free_space_grids, alpha=0.9)

        self.get_logger().info("Current Robot Cell: "+ str(robot_cell))
        self.get_logger().info("Current Robot Rotation: "+ str(robot_rotation))

        #* Storing the calculated free space grid in history ---------------
        # Update past free space grids
        self.free_space_grids.append(free_space_grid)
        if(len(self.free_space_grids) > cfg.WINDOW_SIZE):
            self.free_space_grids.pop(0)

            for i in range(cfg.probability_filter_extent):
                #Remove low probability cells from older grids

                min_value = np.min(self.free_space_grids[i])
                max_value = np.max(self.free_space_grids[i])
                threshold = min_value + (max_value - min_value) * cfg.probability_filter
                self.free_space_grids[i][self.free_space_grids[i] < threshold] = 0

        # Convert grids to OccupancyGrid messages
        occupancy_grid_msg = self.numpy_to_occupancy_grid(
            self.occupancy_grids[-1], cfg.CELL_SIZE / 100, cfg.map_width, cfg.map_height
        )
        free_space_grid_msg = self.numpy_to_occupancy_grid(
            self.free_space_grids[-1], cfg.CELL_SIZE / 100, cfg.map_width, cfg.map_height
        )

        # Publish the messages
        self.occupancy_pub_raw.publish(occupancy_grid_msg)
        self.free_space_pub_raw.publish(free_space_grid_msg)

        # Publish now for Rviz (Flipped and Rotated)
        occupancy_grid_msg_flipped = self.numpy_to_occupancy_grid(
            np.flipud(np.rot90(self.occupancy_grids[-1])), cfg.CELL_SIZE / 100, cfg.map_height, cfg.map_width
        )
        free_space_grid_msg_flipped = self.numpy_to_occupancy_grid(
            np.flipud(np.rot90(self.free_space_grids[-1])), cfg.CELL_SIZE / 100, cfg.map_height, cfg.map_width
        )

        self.occupancy_pub.publish(occupancy_grid_msg_flipped)
        self.free_space_pub.publish(free_space_grid_msg_flipped)

        # Publish the robot pose
        self.br.sendTransform(self.pose_to_transform('map', 'base_link', self.robot_pose))

        self.get_logger().info("Occupancy Grid & Free Space Grid Published")

    def numpy_to_occupancy_grid(self, grid, resolution, width, height, frame_id='map'):
        occupancy_grid = OccupancyGrid()
        occupancy_grid.header = Header()
        occupancy_grid.header.stamp = self.get_clock().now().to_msg()
        occupancy_grid.header.frame_id = frame_id

        occupancy_grid.info.resolution = resolution
        occupancy_grid.info.width = width
        occupancy_grid.info.height = height
        occupancy_grid.info.origin = Pose()
        occupancy_grid.info.origin.position.x = - (width * resolution) / 2
        occupancy_grid.info.origin.position.y = - (height * resolution) / 2
        occupancy_grid.info.origin.position.z = 0.0
        
        occupancy_grid.info.origin.orientation = Quaternion(w=1.0)

        flat_grid = grid.flatten()
        occupancy_grid.data = [int(cell * 100) for cell in flat_grid]  # Assuming grid values are between 0 and 1

        return occupancy_grid
    
    def pose_to_transform(self, parent_frame, child_frame, transform_matrix): #3 x 3 matrix homogeneous
        
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = parent_frame
        t.child_frame_id = child_frame

        # Extract translation
        translation = transform_matrix[:2, 2]
        t.transform.translation.x = translation[0] / 100  # Convert to meters
        t.transform.translation.y = translation[1] / 100  # Convert to meters
        t.transform.translation.z = 0.0  # Assuming 2D transformation

        # Extract rotation
        rotation_matrix = transform_matrix[:2, :2]
        rotation = R.from_matrix(np.vstack([np.hstack([rotation_matrix, [[0], [0]]]), [0, 0, 1]]))
        q = rotation.as_quat()

        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        return t
    
    def transform_to_pose(self, transform):
        
        # pose will be a 3x3 matrix
        pose = np.eye(3)
        # Extract translation
        pose[:2, 2] = [transform.transform.translation.x, transform.transform.translation.y]

        # Extract rotation in radians (w.r.t positive x-axis)
        q = [transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w]
        rotation = R.from_quat(q)
        euler_angles = rotation.as_euler('xyz')
        pose[:2, :2] = rotation.as_matrix()[:2, :2]

        return pose

    def xyz_to_transform(self, parent_frame, child_frame, x, y, z, roll, pitch, yaw):
        
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = parent_frame
        t.child_frame_id = child_frame

        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = z

        q = R.from_euler('xyz', [roll, pitch, yaw]).as_quat()
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        return t

    def update_pose(self, msg):

        delta_pose = np.array(msg.data).reshape((3, 3))
        self.get_logger().info(f'Updated pose:\n{delta_pose}')
        self.robot_pose = np.dot(self.robot_pose, delta_pose)
        return self.robot_pose

def main(args=None):
    rclpy.init(args=args)
    update_grids = UpdateGrids()
    rclpy.spin(update_grids)
    update_grids.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
