from collections import deque
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


# A queue node used in BFS
class Node:
    # (x, y) represents coordinates of a cell in the matrix
    # maintain a parent node for the printing path
    def __init__(self, x, y, parent=None):
        self.x = x
        self.y = y
        self.parent = parent

    def __repr__(self):
        return str((self.x, self.y))

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y


# Below lists detail all four possible movements from a cell

def get_line(path):
    x_coordinates = []
    y_coordinates = []

    for node in path:
        x_coordinates.append(node.x)
        y_coordinates.append(node.y)
    return x_coordinates, y_coordinates


# The function returns false if (x, y) is not a valid position
def isValid(x, y, N):
    # check if this point in the space of obstacles or not
    condition1 = (y >= first_obstacle[0][1] and x <= first_obstacle[1][0])
    condition2 = (second_obstacle[0][0] <= x <= second_obstacle[2][0] and y <= second_obstacle[1][1])
    condition3 = (third_obstacle[0][0] <= x <= third_obstacle[1][0] and y >= third_obstacle[0][1])

    if condition1 or condition3 or condition2:
        return False

    return (0 <= x < N) and (0 <= y < N)


# Utility function to find path from source to destination
def getPath(node, path=[]):
    while node:
        path.append(node)
        node = node.parent


# Find the shortest route in a matrix from source cell (x, y) to
# destination cell (dist_x, dist_y)
def findPath(matrix, x, y, dist_x, dist_y):
    # base case
    if not matrix or not len(matrix):
        return

    # `N × N` matrix
    N = len(matrix)

    # create a queue and enqueue the first node
    q = deque()
    src = Node(x, y)
    q.append(src)

    # set to check if the matrix cell is visited before or not
    visited = set()

    key = (src.x, src.y)
    visited.add(key)

    # loop till queue is empty
    while q:

        # dequeue front node and process it
        curr = q.popleft()
        i = curr.x
        j = curr.y

        # return if the destination is found
        if i == dist_x and j == dist_y:
            path = []
            getPath(curr, path)
            return path

        # value of the current cell
        n = matrix[i][j]

        # check all four possible movements from the current cell
        # and recur for each valid movement
        for k in range(len(row)):
            # get next position coordinates using the value of the current cell
            x = i + row[k] * n
            y = j + col[k] * n

            # check if it is possible to go to the next position
            # from the current position
            if isValid(x, y, N):
                # construct the next cell node
                next = Node(x, y, curr)
                key = (next.x, next.y)

                # if it isn't visited yet
                if key not in visited:
                    # enqueue it and mark it as visited
                    q.append(next)
                    visited.add(key)

    # return None if the path is not possible
    return


def draw_path():
    fig, ax = plt.subplots(1, figsize=(8, 6))
    fig.suptitle('The Shortest Path')

    # Plot the data

    plt.xlim(0, 1000)
    plt.ylim(0, 1000)
    # Show the major grid lines with dark grey lines
    plt.grid(visible=True, which='major', color='#666666')

    # Show the minor grid lines with very faint and almost transparent grey lines
    plt.minorticks_on()
    plt.grid(visible=True, which='minor', color='#999999')

    # draw the obstacles
    rect1 = patches.Rectangle((first_obstacle[0]), 200, 200, facecolor='grey')
    rect2 = patches.Rectangle((second_obstacle[0]), 200, 800, facecolor='grey')
    rect3 = patches.Rectangle((third_obstacle[0]), 100, 400, facecolor='grey')

    # add obstacles to the grid
    ax.add_patch(rect1)
    ax.add_patch(rect2)
    ax.add_patch(rect3)

    # get points of line
    x_coordinates, y_coordinates = get_line(path)

    plt.scatter(x_coordinates, y_coordinates)
    plt.plot(x_coordinates, y_coordinates)

    # draw the start and end positions of the robot
    circle = patches.Circle((path[0].x, path[0].y), 10, color="black")
    circle1 = patches.Circle((path[-1].x, path[-1].y), 10, color="black")

    ax.add_patch(circle)
    ax.add_patch(circle1)

    plt.show()


if __name__ == '__main__':
    # the possible movement
    row = [-10, 0, 0, 10]
    col = [0, -10, 10, 0]

    # create the obstacles
    first_obstacle = np.array([[0, 800], [200, 800], [0, 1000], [200, 1000]])
    second_obstacle = np.array([[400, 0], [400, 800], [600, 0], [600, 800]])
    third_obstacle = np.array([[700, 600], [800, 600], [700, 1000], [800, 1000]])

    # create the grid (robot space)
    matrix = np.ones(1000 * 1000).reshape(1000, 1000)
    matrix = matrix.astype(int)

    grid = []
    # convert numpy array to list 
    for i in matrix:
        grid.append(list(i))

    path = findPath(grid, 10, 10, 950, 950)

    if path:
        path.reverse()
        print('The shortest path is', path)
        draw_path()
    else:
        print('Destination is not found')
