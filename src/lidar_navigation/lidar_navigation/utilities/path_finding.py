
#! A* functions ----------------------------------------------------

# if a cell is given, return the 4-connected neighbors of the cell
def get_neighbors(grid, cell, map_height, map_width):

    neighbors = []
    x, y = cell

    # Check if the neighbor is within the map and is free
    if x > 0 and grid[x - 1,y] != 0:
        neighbors.append((x - 1, y))
    if x < map_width - 1 and grid[x + 1,y] != 0:
        neighbors.append((x + 1, y))
    if y > 0 and grid[x, y - 1] != 0:
        neighbors.append((x, y - 1))
    if y < map_height - 1 and grid[x, y + 1] != 0:
        neighbors.append((x, y + 1))
        
    if x > 0 and y > 0 and grid[x - 1, y - 1] != 0:
        neighbors.append((x - 1, y - 1))
    if x < map_width - 1 and y > 0 and grid[x + 1, y - 1] != 0:
        neighbors.append((x + 1, y - 1))
    if x > 0 and y < map_height - 1 and grid[x - 1, y + 1] != 0:
        neighbors.append((x - 1, y + 1))
    if x < map_width - 1 and y < map_height - 1 and grid[x + 1, y + 1] != 0:
        neighbors.append((x + 1, y + 1))

    return neighbors

# depends on the probability of the cell being free + the direction of movement
# probability  high -----> cost low -----> high priority
def cost(current_cell, neighbor, free_space_grid):

    if(current_cell.parent):
        parent = current_cell.parent.value
        current = current_cell.value
        #check if direction has not changed
        direction1 = (parent[0] - current[0], parent[1] - current[1])
        direction2 = (current[0] - neighbor[0], current[1] - neighbor[1])

    else:
        parent = current_cell.value
        current = current_cell.value
        #make direction 90 degrees
        direction1 = (0, 1)
        direction2 = (current[0] - neighbor[0], current[1] - neighbor[1])

    if direction1 == direction2:
        penalty = 0.1
    else:
        penalty = 1

    return penalty*0.9 + round(1 - free_space_grid[neighbor],2)*0.1

# approximate distance between two cells ( Start ----> End )
def heuristic(current_cell, end_cell):
    return abs(current_cell[0] - end_cell[0]) + abs(current_cell[1] - end_cell[1])

#! A* algorithm ----------------------------------------------------

class node:

    childs = []
    parent = None
    cost = 0
    name = ""
    value = None

    #Constructor
    def __init__(self, value=None, childs=[], parent=None, cost=0, name=None):

        self.childs = childs
        self.value = value
        self.name = name
        self.parent = parent
        self.cost = cost

    #print function / string representation
    def __repr__(self) -> str:

        if(self.name):
            return self.name
        else:
            return str(self.value)
        
    #equality function
    def __eq__(self, o: object) -> bool:
        return self.value == o.value

# Calculate the path from the start cell to the end cell
# [(x1, y1), (x2, y2), ... , (xn, yn)]
def astar(start_cell, end_cell, grid, map_height, map_width):

    unvisited = []
    visited = []

    # Create the start node
    start_node = node(start_cell, cost=0)

    # Add the start node to the unvisited list
    unvisited.append(start_node)

    while unvisited:

        # Get the node with the minimum cost
        current_node = min(unvisited, key=lambda x: x.cost)

        # Remove the current node from the unvisited list
        unvisited.remove(current_node)

        # Add the current node to the visited list
        visited.append(current_node)

        # Check if the current node is the end node
        if current_node.value == end_cell:
            path = []
            while current_node:
                path.append(current_node.value)
                current_node = current_node.parent
            return path[::-1], simplify_path(path[::-1])

        # Get the neighbors of the current node
        neighbors = get_neighbors(grid, current_node.value, map_height, map_width)

        for neighbor in neighbors:

            # Create the neighbor node
            neighbor_node = node(neighbor, parent=current_node, cost=current_node.cost + cost(current_node, neighbor, grid) + heuristic(neighbor, end_cell))

            # Check if the neighbor is in the visited list
            if neighbor_node in visited:
                continue

            # Check if the neighbor is in the unvisited list
            if neighbor_node not in unvisited:
                unvisited.append(neighbor_node)
            else:
                # Replace the neighbor node with the new node if the new node has a lower cost
                for node_ in unvisited:
                    if node_ == neighbor_node and node_.cost > neighbor_node.cost:
                        node_ = neighbor_node
    return None, None

#! Movement functions ----------------------------------------------

def simplify_path(path):
    if not path:
        return []

    simplified_path = []
    current_direction = None

    for i in range(1, len(path)):
        prev_cell = path[i - 1]
        current_cell = path[i]

        # Calculate the direction of movement
        direction = (current_cell[0] - prev_cell[0], current_cell[1] - prev_cell[1])

        if direction != current_direction:
            simplified_path.append(prev_cell)
            current_direction = direction

    simplified_path.append(path[-1])
    
    return simplified_path
