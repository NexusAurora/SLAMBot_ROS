import numpy as np

#! Lider Constants -----------------

FOV_DEGREES = 101
NUM_READINGS = 101
START_DEGREE = 40
END_DEGREE = 140
CELL_SIZE = 2
WINDOW_SIZE = 5

#Error Corrections [in cm]
fix_error = 0
servo_offset = 2.8
body_offset_y = 8
min_distance = 3
max_distance = 100

# Initialize an empty occupancy grid map
map_width = 107//(CELL_SIZE - 1)
map_height = 107//(CELL_SIZE - 1)

probability_filter = 0.2
probability_filter_extent = 2 #! Should be less than WINDOW_SIZE

erotion_extent = 2
erosion_kernel = np.ones((3, 3), np.uint8)
erosion_contritubion = 0.05

#ICP
icp_iterations = 1
icp_tolerance = 0.0001

#display
display_scale = 5
guide_color = (225, 86, 43)

#! Scans ---------------------------

robot_pose = (0, 0, 0)  # (x, y, theta)

end_coordinate = (0, 30)
end_cell = (0, 0)

#! Path ----------------------------

path = None
simplified_path = None

#! Movement ------------------------

path_wait_no = 10 # Number of times to wait for a path updates before generating commands

wait_time = 20  #seconds - waiting for the robot to reach the target
wait_to_send = 1 #wait time to send data
