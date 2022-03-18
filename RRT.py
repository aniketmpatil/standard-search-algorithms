# Standard Algorithm Implementation
# Sampling-based Algorithms RRT and RRT*

import matplotlib.pyplot as plt
import numpy as np
import math
from scipy import spatial

EXTEND_DIST = 10
GOAL_BIAS = 0.05
GOAL_DIST = 15
COLLISION_STEPS = 50

RRTS_NEIGHBORS = 20

# Class for each tree node
class Node:
    def __init__(self, row, col):
        self.row = row        # coordinate
        self.col = col        # coordinate
        self.parent = None    # parent node
        self.cost = 0.0       # cost


# Class for RRT
class RRT:
    # Constructor
    def __init__(self, map_array, start, goal):
        self.map_array = map_array            # map array, 1->free, 0->obstacle
        self.size_row = map_array.shape[0]    # map size
        self.size_col = map_array.shape[1]    # map size

        self.start = Node(start[0], start[1]) # start node
        self.goal = Node(goal[0], goal[1])    # goal node
        self.vertices = []                    # list of nodes
        self.found = False                    # found flag
        

    def init_map(self):
        '''Intialize the map before each search
        '''
        self.found = False
        self.goal.cost = math.inf
        self.vertices = []
        self.vertices.append(self.start)

    
    def dis(self, node1, node2):
        '''Calculate the euclidean distance between two nodes
        arguments:
            node1 - node 1
            node2 - node 2

        return:
            euclidean distance between two nodes
        '''
        dist = math.sqrt(math.pow((node1.row - node2.row), 2) + math.pow((node1.col - node2.col), 2))
        return dist

    
    def check_collision(self, node1, node2):
        '''Check if the path between two nodes collide with obstacles
        arguments:
            node1 - node 1
            node2 - node 2

        return:
            True if the new node is valid to be connected
        '''
        ### YOUR CODE HERE ###
        dx = node2.row - node1.row
        dy = node2.col - node1.col
        divN = 1/COLLISION_STEPS
        xstep = dx * divN
        ystep = dy * divN

        xpt = node1.row
        ypt = node1.col
        for i in range(COLLISION_STEPS):
            if(self.map_array[int(xpt)][int(ypt)] == 0):
                return False
            xpt = xpt + xstep
            ypt = ypt + ystep
        return True


    def get_new_point(self, goal_bias):
        '''Choose the goal or generate a random point
        arguments:
            goal_bias - the possibility of choosing the goal instead of a random point

        return:
            point - the new point
        '''
        ### YOUR CODE HERE ###
        random_node = Node(np.random.randint(0, self.size_row), np.random.randint(0, self.size_col))
        point = np.random.choice([self.goal, random_node], p=[goal_bias, 1 - goal_bias])
        return point

    
    def get_nearest_node(self, point):
        '''Find the nearest node in self.vertices with respect to the new point
        arguments:
            point - the new point

        return:
            the nearest node
        '''
        ### YOUR CODE HERE ###
        min_dist = math.inf
        for vertex in self.vertices:
            if(self.dis(vertex, point) < min_dist):
                min_dist = self.dis(vertex, point)
                nearest_node = vertex
        return nearest_node


    def extend(self, node1, node2):
        '''Find the node to be added to the tree, returns either node2 or a new node
        in the direction of node2 based on a distance comparison
        arguments:
            node1 - node 1
            node2 - node 2

        return:
            node2 is distance is too short for extension
            new_node from node1, in the direction of node2
        '''
        if(self.dis(node1, node2) <= EXTEND_DIST) and \
            (node2.row != self.goal.row) and (node2.col != self.goal.col):
            # Checks that the extend goal is not the goal node to avoid goal.parent being set as goal
            return node2
        else:
            dx = node2.row - node1.row
            dy = node2.col - node1.col
            mod = self.dis(node1, node2)
            stepx = dx * EXTEND_DIST / mod
            stepy = dy * EXTEND_DIST / mod
            x = node1.row + stepx
            y = node1.col + stepy

            # Boundary check
            if(x < 0): x = 0
            elif(x > self.size_row): x = self.size_row - 1
            if(y < 0): y = 0
            elif(y > self.size_col): y = self.size_col - 1

            new_node = Node(x, y)
            new_node.parent = node1
            new_node.cost = node1.cost + self.dis(new_node, node1)
            return new_node


    def get_neighbors(self, new_node, neighbor_size):
        '''Get the neighbors that are within the neighbor distance from the node
        arguments:
            new_node - a new node
            neighbor_size - the neighbor distance

        return:
            neighbors - a list of neighbors that are within the neighbor distance 
        '''
        ### YOUR CODE HERE ###
        neighbors = []
        for vertex in self.vertices:
            if(self.dis(vertex, new_node) < neighbor_size):
                neighbors.append(vertex)
        return neighbors


    def rewire(self, new_node, neighbors):
        '''Rewire the new node and all its neighbors
        arguments:
            new_node - the new node
            neighbors - a list of neighbors that are within the neighbor distance from the node

        Rewire the new node if connecting to a new neighbor node will give least cost.
        Rewire all the other neighbor nodes.
        '''
        ### YOUR CODE HERE ###
        for neighbor in neighbors:
            new_cost = new_node.cost + self.dis(neighbor, new_node)
            if(new_cost < neighbor.cost and self.check_collision(new_node, neighbor)):
                neighbor.cost = new_cost
                neighbor.parent = new_node

    
    def draw_map(self):
        '''Visualization of the result
        '''
        # Create empty map
        fig, ax = plt.subplots(1)
        img = 255 * np.dstack((self.map_array, self.map_array, self.map_array))
        ax.imshow(img)

        # Draw Trees or Sample points
        for node in self.vertices[1:-1]:
            plt.plot(node.col, node.row, markersize=3, marker='o', color='y')
            plt.plot([node.col, node.parent.col], [node.row, node.parent.row], color='y')
        
        # Draw Final Path if found
        if self.found:
            cur = self.goal
            while cur.col != self.start.col or cur.row != self.start.row:
                plt.plot([cur.col, cur.parent.col], [cur.row, cur.parent.row], color='b')
                cur = cur.parent
                plt.plot(cur.col, cur.row, markersize=3, marker='o', color='b')

        # Draw start and goal
        plt.plot(self.start.col, self.start.row, markersize=5, marker='o', color='g')
        plt.plot(self.goal.col, self.goal.row, markersize=5, marker='o', color='r')

        # show image
        plt.show()


    def RRT(self, n_pts=1000):
        '''RRT main search function
        arguments:
            n_pts - number of points try to sample, 
                    not the number of final sampled points

        In each step, extend a new node if possible, and check if reached the goal
        '''
        # Remove previous result
        self.init_map()
        print(" --------- RRT Algorithm ---------- ")

        ### YOUR CODE HERE ###

        # In each step,
        # get a new point, 
        # get its nearest node, 
        # extend the node and check collision to decide whether to add or drop,
        # if added, check if reach the neighbor region of the goal.
        
        for n in range(n_pts):
            # Sample new point and get its nearest neighbor in the tree
            new_point = self.get_new_point(GOAL_BIAS)
            near_vertex = self.get_nearest_node(new_point)

            # Extend in the direction of the random node if no collision
            step_node = self.extend(near_vertex, new_point)
            if(self.check_collision(near_vertex, step_node)):
                step_node.parent = near_vertex
                step_node.cost = near_vertex.cost + self.dis(step_node, near_vertex)
                self.vertices.append(step_node)
        
            # Check if goal node is also within close range (GOAL_DIST) and add if no collision
            # ========== First Method ========== 
            # Stop exploring the tree once the goal node is reached

            if((self.dis(step_node, self.goal) <= GOAL_DIST) and \
                self.check_collision(step_node, self.goal)):
                self.found = True
                self.goal.parent = step_node
                self.goal.cost = step_node.cost + self.dis(step_node, self.goal)
                break

            # ========== Second Method ========== 
            # Check for neighbors of goal node and connect if there's a neighbor with lower cost than current goal cost
            # This method keeps exploring the tree, even if goal is reached to find a better path to the goal node

            # goal_neighbors = self.get_neighbors(self.goal, GOAL_DIST)
            # for neighbor in goal_neighbors:
            #     if(self.check_collision(neighbor, self.goal) and \
            #         (neighbor.cost + self.dis(neighbor, self.goal)) < self.goal.cost):
            #         self.goal.parent = neighbor
            #         self.goal.cost = neighbor.cost + self.dis(neighbor, self.goal)
            #         self.found = True

        # Output
        if self.found:
            self.vertices.append(self.goal)
            steps = len(self.vertices) - 2
            length = self.goal.cost
            print("It took %d nodes to find the current path" %steps)
            print("The path length is %.2f" %length)
        else:
            print("No path found")
        
        print(" -------------------------------- ")
        
        # Draw result
        self.draw_map()


    def RRT_star(self, n_pts=1000, neighbor_size=20):
        '''RRT* search function
        arguments:
            n_pts - number of points try to sample, 
                    not the number of final sampled points
            neighbor_size - the neighbor distance
        
        In each step, extend a new node if possible, and rewire the node and its neighbors
        '''
        # Remove previous result
        self.init_map()
        print(" --------- RRT* Algorithm ---------- ")

        ### YOUR CODE HERE ###

        # In each step,
        # get a new point, 
        # get its nearest node, 
        # extend the node and check collision to decide whether to add or drop,
        # if added, rewire the node and its neighbors,
        # and check if reach the neighbor region of the goal if the path is not found.

        for n in range(n_pts):
            # Sample new point and get its nearest neighbor in the tree
            new_point = self.get_new_point(GOAL_BIAS)
            near_vertex = self.get_nearest_node(new_point)

            # Find the node to extend in the direction of new node
            step_node = self.extend(near_vertex, new_point)
            if(self.check_collision(near_vertex, step_node)):
                neighbors = self.get_neighbors(step_node, neighbor_size)
                min_node = near_vertex
                min_cost = near_vertex.cost + self.dis(near_vertex, step_node)
                for neighbor in neighbors:
                    if(self.check_collision(neighbor, step_node) and \
                        (neighbor.cost + self.dis(neighbor, step_node)) < min_cost):
                        min_node = neighbor
                        min_cost = neighbor.cost + self.dis(neighbor, step_node)
                step_node.parent = min_node
                step_node.cost = min_cost
                self.vertices.append(step_node)
                self.rewire(step_node, neighbors)

            # Check for neighbors of goal node and connect if there's a neighbor with lower cost than current goal cost
            # This method keeps exploring the tree, even if goal is reached to find a better path to the goal node
            goal_neighbors = self.get_neighbors(self.goal, neighbor_size)
            for neighbor in goal_neighbors:
                if(self.check_collision(neighbor, self.goal) and \
                    (neighbor.cost + self.dis(neighbor, self.goal)) < self.goal.cost):
                    self.goal.parent = neighbor
                    self.goal.cost = neighbor.cost + self.dis(neighbor, self.goal)
                    self.found = True

        # Output
        if self.found:
            self.vertices.append(self.goal)
            steps = len(self.vertices) - 2
            length = self.goal.cost
            print("It took %d nodes to find the current path" %steps)
            print("The path length is %.2f" %length)
        else:
            print("No path found")
        
        print(" -------------------------------- ")

        # Draw result
        self.draw_map()
