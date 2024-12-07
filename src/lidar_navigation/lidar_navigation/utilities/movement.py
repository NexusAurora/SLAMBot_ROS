import numpy as np

def forward(distance):

    '''
    Input: distance (float) - distance from the origin
    Output: Time (float) - time to tuen on motors (payload)    
    '''

    weights = np.load('movement_weights.npy', allow_pickle=True).item()
    #return a * x**3 + b * x**2 + c * x + d
    time = weights['forward'][0] * distance ** 3 + weights['forward'][1] * distance ** 2 + weights['forward'][2] * distance + weights['forward'][3]
    #round to integer
    time = round(time)
    #print("weights: ", weights['forward'])

    return {
        "d": 0,
        "t": time,
        "s": 100
    }

def backward(distance):

    '''
    Input: distance (float) - distance from the origin
    Output: Time (float) - time to tuen on motors (payload)
    '''

    weights = np.load('movement_weights.npy', allow_pickle=True).item()
    
    time = weights['backward'][0] * distance ** 3 + weights['backward'][1] * distance ** 2 + weights['backward'][2] * distance + weights['backward'][3]
    #round to integer
    time = round(time)
    #print("weights: ", weights['backward'])

    return {
        "d": 1,
        "t": time,
        "s": 100
    }

def left(degree):

    '''
    Input: distance (float) - degree wtr the origin
    Output: Time (float) - time to tuen on motors (payload)
    '''

    weights = np.load('movement_weights.npy', allow_pickle=True).item()

    time = weights['left'][0] * degree ** 3 + weights['left'][1] * degree ** 2 + weights['left'][2] * degree + weights['left'][3]
    #round to integer
    time = round(time)
    #print("weights: ", weights['left'])

    return {
        "d": 2,
        "t": time,
        "s": 200
    }

def right(degree):

    '''
    Input: distance (float) - degree wtr the origin
    Output: Time (float) - time to tuen on motors (payload)
    '''

    weights = np.load('movement_weights.npy', allow_pickle=True).item()

    time = weights['right'][0] * degree ** 3 + weights['right'][1] * degree ** 2 + weights['right'][2] * degree + weights['right'][3]
    #round to integer
    time = round(time)
    #print("weights: ", weights['right'])

    return {
        "d": 3,
        "t": time,
        "s": 200
    }

def make_init_pose(x, y, theta):
    
        '''
        Input: x (float) - x-coordinate of the robot
            y (float) - y-coordinate of the robot
            theta (float) - angle of the robot wtr the x-axis
        Output: init_pose (np.array) - initial pose of the robot
        '''
    
        # make the initial pose
        init_pose = np.identity(3)
        init_pose[0, 2] = x
        init_pose[1, 2] = y
        init_pose[0, 0] = np.cos(np.deg2rad(theta))
        init_pose[0, 1] = -np.sin(np.deg2rad(theta))
        init_pose[1, 0] = np.sin(np.deg2rad(theta))
        init_pose[1, 1] = np.cos(np.deg2rad(theta))
    
        return init_pose

def generate_movement_commands(simplified_path, robot_rotation, angle_tolerance=10, distance_tolerance=10):

    if(len(simplified_path) < 2):
        return None, None
    
    print("\nMoving ----------------------------------")
    print("simplified_path: ", simplified_path)

    start_cell = simplified_path[0]
    end_cell = simplified_path[1]
    goal_cell = simplified_path[-1]

    print("start_cell: ", start_cell)
    print("end_cell: ", end_cell)

    # target rotation
    target_rotation = np.arctan2(end_cell[1] - start_cell[1], end_cell[0] - start_cell[0]) - np.pi / 2
    target_rotation = np.rad2deg(target_rotation)

    print("robot_rotation: ", robot_rotation)
    print("target_rotation: ", target_rotation)

    # Calculate the angle to turn
    angle = target_rotation - robot_rotation #angle w.r.t the positive x-axis
    print("angle to turn: ", angle)

    # Calculate the distance to travel
    distance = ((end_cell[0] - start_cell[0]) ** 2 + (end_cell[1] - start_cell[1]) ** 2) ** 0.5

    #total distance
    total_distance = ((goal_cell[0] - start_cell[0]) ** 2 + (goal_cell[1] - start_cell[1]) ** 2) ** 0.5

    print("distance: ", distance)
    print("total_distance: ", total_distance)
    print("-----------------------------------------\n")

    # Determine the direction of movement (angle is w.r.t the positive x-axis)
    if angle > angle_tolerance and (total_distance > distance_tolerance or total_distance < -distance_tolerance):
        command = left(abs(angle_tolerance))
        #init_pose w.r.t previous pose
        init_pose = make_init_pose(0, 0, angle_tolerance)
    elif angle < -angle_tolerance and (total_distance > distance_tolerance or total_distance < -distance_tolerance):
        command = right(abs(angle_tolerance))
        #init_pose w.r.t previous pose
        init_pose = make_init_pose(0, 0, -angle_tolerance)
    elif distance > 0 and (total_distance > distance_tolerance or total_distance < -distance_tolerance):
        command = forward(distance)
        #init_pose w.r.t previous pose
        init_pose = make_init_pose(end_cell[0] - start_cell[0], end_cell[1] - start_cell[1], 0)
    elif distance < 0 and (total_distance > distance_tolerance or total_distance < -distance_tolerance):
        command = backward(abs(distance))
        #init_pose w.r.t previous pose
        init_pose = make_init_pose(end_cell[0] - start_cell[0], end_cell[1] - start_cell[1], 0)
    else:
        command = None
        init_pose = None
    
    return command, init_pose
