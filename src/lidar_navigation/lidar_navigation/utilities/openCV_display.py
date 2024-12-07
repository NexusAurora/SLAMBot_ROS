import cv2
import numpy as np

def add_grid_lines_and_markings(frame, map_width, map_height, display_scale, guide_color):
    """
    Adds grid lines and grid markings to the given frame.

    Parameters:
    - frame: The input frame to which grid lines and markings will be added.
    - map_width: The width of the map.
    - map_height: The height of the map.
    - display_scale: The scale factor for displaying the map.
    - guide_color: The color of the grid lines and markings.

    Returns:
    - The frame with grid lines and markings added.
    """
    # Add grid lines
    for i in range(1, map_width):
        cv2.line(frame, (i * display_scale, 0), (i * display_scale, map_height * display_scale), guide_color, 1)
    for i in range(1, map_height):
        cv2.line(frame, (0, i * display_scale), (map_width * display_scale, i * display_scale), guide_color, 1)

    # Adjust grid markings from -half to +half
    half_width = map_width // 2
    half_height = map_height // 2

    # Add grid markings for width
    for i in range(-half_width, half_width, 5):
        # Calculate the correct position for the marking
        pos_x = (i + half_width) * display_scale
        # Display the marking, adjusting the position to start from -half_width
        cv2.putText(frame, str(i), (pos_x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, guide_color, 1)

    # Add grid markings for height
    for i in range(-half_height, half_height, 5):
        # Calculate the correct position for the marking
        pos_y = (i + half_height) * display_scale
        # Display the marking, adjusting the position to start from -half_height
        cv2.putText(frame, str(i), (10, pos_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, guide_color, 1)
    
    return frame

def process_frame(frame,colormap,robot_cell=None, robot_rotation=None, display_scale=8, guide_color=(225, 86, 43)):

    '''
    Input --> 2D Matrix (values 0 - 1)
    Output --> 2D Matrix (values 0 - 255) with grid lines
    '''

    #* Normalize frame and convert to proper values (0 -255) -------------------------------------
    # Normalize the frame to 0-255 for proper grayscale display
    # 0 - 1 to 0 - 255
    frame_normalized = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)

    #* Other Image processing ---------------------------------------------------------------------
    # Convert to uint8 type for imshow
    frame_normalized = np.uint8(frame_normalized)

    # Apply the colormap
    frame_normalized = cv2.applyColorMap(frame_normalized, colormap)

    #* Marking robot's position and orientation by a arrow / line -------------------------------------
    #add robot cell
    if(robot_cell):
        robot_pose_grid_x, robot_pose_grid_y = robot_cell
        frame_normalized[robot_pose_grid_x, robot_pose_grid_y] = [0, 0, 255]
        
        '''
        #draw a small arrow to indicate the robot orientation
        if(robot_rotation):
            robot_rotation = robot_rotation + np.pi/2
            arrow_length = 5
            arrow_x = int(arrow_length * np.sin(robot_rotation))
            arrow_y = int(arrow_length * np.cos(robot_rotation))
            cv2.arrowedLine(frame_normalized, (robot_pose_grid_y, robot_pose_grid_x), (robot_pose_grid_y + arrow_y, robot_pose_grid_x + arrow_x), (0, 0, 255), 1)
        '''
    # Flip the frame vertically
    #frame_flipped = cv2.flip(frame_normalized, 0)
    
    #rotate 90 degrees counter clockwise
    frame_flipped = cv2.rotate(frame_normalized, cv2.ROTATE_90_COUNTERCLOCKWISE)

    #* Adding Grid Lines and Resizing ---------------------------------------------------------------
    #resize
    frame_flipped = cv2.resize(frame_flipped, (frame.shape[1] * display_scale, frame.shape[0] * display_scale), interpolation=cv2.INTER_NEAREST)
    
    # Add grid lines and markings
    frame_flipped = add_grid_lines_and_markings(frame_flipped, frame.shape[1], frame.shape[0], display_scale, guide_color)

    return frame_flipped

def process_frame_path(frame, colormap, robot_cell=None, robot_rotation=None, end_cell=None, path = None, display_scale=8, guide_color=(225, 86, 43)):

    '''
    Input --> 2D Matrix (values 0 - 1)
    Output --> 2D Matrix (values 0 - 255) with grid lines
    '''
    # Normalize the frame to 0-255 for proper grayscale display
    frame_normalized = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)

    # Convert to uint8 type for imshow
    frame_normalized = np.uint8(frame_normalized)

    # Apply the colormap
    frame_normalized = cv2.applyColorMap(frame_normalized, colormap)

    #add path as lines (Grey color)
    if(path):
        for i in range(len(path) - 1):
            cell1_x, cell1_y = path[i]
            cell2_x, cell2_y = path[i + 1]
            cv2.line(frame_normalized, (cell1_y, cell1_x), (cell2_y, cell2_x), (128, 128, 128), 1)

    #add path as points
    if(path):
        for cell in path:
            cell_x, cell_y = cell
            frame_normalized[cell_x, cell_y] = [255, 255, 255]

    #add robot cell
    if(robot_cell):
        robot_pose_grid_x, robot_pose_grid_y = robot_cell
        frame_normalized[robot_pose_grid_x, robot_pose_grid_y] = [0, 0, 255]
        
        '''#draw a small arrow to indicate the robot orientation
        if(robot_rotation):
            robot_rotation = robot_rotation + np.pi/2
            arrow_length = 5
            arrow_x = int(arrow_length * np.sin(robot_rotation))
            arrow_y = int(arrow_length * np.cos(robot_rotation))
            cv2.arrowedLine(frame_normalized, (robot_pose_grid_x, robot_pose_grid_y), (robot_pose_grid_x + arrow_x, robot_pose_grid_y + arrow_y), (0, 0, 255), 1)
    '''
    #add end cell
    if(end_cell):
        end_pose_grid_x, end_pose_grid_y = end_cell
        #purple dot (goal)
        frame_normalized[end_pose_grid_x, end_pose_grid_y] = [255, 0, 255]

    #rotate 90 degrees counter clockwise
    frame_flipped = cv2.rotate(frame_normalized, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    #resize
    frame_flipped = cv2.resize(frame_flipped, (frame.shape[1] * display_scale, frame.shape[0] * display_scale), interpolation=cv2.INTER_NEAREST)
    
    # Add grid lines and markings
    frame_flipped = add_grid_lines_and_markings(frame_flipped, frame.shape[1], frame.shape[0], display_scale, guide_color)

    return frame_flipped