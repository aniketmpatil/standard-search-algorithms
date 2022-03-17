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
        if(self.dis(node1, node2) <= EXTEND_DIST):
            return node2
        else:
            dx = node2.row - node1.row
            dy = node2.col - node1.col
            mod = self.dis(node1, node2)
            stepx = dx * EXTEND_DIST / mod
            stepy = dy * EXTEND_DIST / mod
            x = node1.row + stepx
            y = node1.col + stepy

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
            new_cost = neighbor.cost + self.dis(neighbor, new_node)
            if(new_node.cost > new_cost):
                new_node.parent = neighbor
                new_node.cost = new_cost

    
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

        ### YOUR CODE HERE ###

        # In each step,
        # get a new point, 
        # get its nearest node, 
        # extend the node and check collision to decide whether to add or drop,
        # if added, check if reach the neighbor region of the goal.
        
        for n in range(n_pts):
            new_point = self.get_new_point(GOAL_BIAS)
            near_vertex = self.get_nearest_node(new_point)
            add_node = self.extend(near_vertex, new_point)
            if(self.check_collision(near_vertex, add_node)):
                add_node.parent = near_vertex
                add_node.cost = near_vertex.cost + self.dis(add_node, near_vertex)
                self.vertices.append(add_node)
            if((self.dis(add_node, self.goal) <= GOAL_DIST) and \
                self.check_collision(add_node, self.goal)):
                self.found = True
                self.goal.parent = add_node
                self.goal.cost = add_node.cost + self.dis(add_node, self.goal)
                self.vertices.append(self.goal)
                break

        # Output
        if self.found:
            steps = len(self.vertices) - 2
            length = self.goal.cost
            print("It took %d nodes to find the current path" %steps)
            print("The path length is %.2f" %length)
        else:
            print("No path found")
        
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

        ### YOUR CODE HERE ###

        # In each step,
        # get a new point, 
        # get its nearest node, 
        # extend the node and check collision to decide whether to add or drop,
        # if added, rewire the node and its neighbors,
        # and check if reach the neighbor region of the goal if the path is not found.
        GOAL_DIST = 10
        for n in range(n_pts):
            new_node = self.get_new_point(0.05)
            near_vertex = self.get_nearest_node(new_node)
            if(self.check_collision(near_vertex, new_node)):
                continue
            self.vertices.append(new_node)
            new_node.parent = near_vertex
            new_node.cost = self.dis(new_node, near_vertex)
            print(self.dis(new_node, self.goal))
            if((self.dis(new_node, self.goal) <= GOAL_DIST) and \
                self.check_collision(new_node, self.goal)):
                print("True")
                self.found = True
                self.goal.cost = self.dis(new_node, self.goal)
                self.vertices.append(self.goal)
                break

        # Output
        if self.found:
            steps = len(self.vertices) - 2
            length = self.goal.cost
            print("It took %d nodes to find the current path" %steps)
            print("The path length is %.2f" %length)
        else:
            print("No path found")

        # Draw result
        self.draw_map()
