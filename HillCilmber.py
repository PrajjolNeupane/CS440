import random
import math
class hillCimber():
    def __init__(self, enviorment):
        self.matrix = enviorment

    def two_dimensional_hill_climbing(self, sim_ann=False, temputare=10.0, cooling_rate=0.5):
        current_position = self.get_start()
        path = [current_position]
        best_score = self.get_score(current_position)
        while True:
            neighbors = self.get_neighbors(current_position)
            best_neighbor = None
            best_neighbor_score = best_score
            for neighbor in neighbors:
                score = self.get_score(neighbor)
                if score > best_neighbor_score:
                    best_neighbor_score = score
                    best_neighbor = neighbor
                elif (score > -math.inf) and sim_ann and random.random() < math.exp((score - best_neighbor_score) / temputare):
                    best_neighbor_score = score
                    best_neighbor = neighbor

            if best_neighbor == None:
                break
            current_position = best_neighbor
            best_score = best_neighbor_score
            path.append(current_position)
            temputare *= cooling_rate
        return path, best_score

    def get_score(self, current_position):
        score = self.matrix[current_position[0]][current_position[1]]
        if isinstance(score, str):
            score = -math.inf
        return score

    def get_start(self):
        for i in range(len(self.matrix)):
            for j in range(len(self.matrix[0])):
                if self.matrix[i][j] == "start":
                    return (i,j)
        return None

    def get_neighbors(self, current_position):
        i = current_position[0]
        j = current_position[1]
        matrix_size = len(self.matrix)
        neighbors = []
        if j - 1 > -1:
            neighbors.append((i, j - 1))
        if i - 1 > -1:
            neighbors.append((i - 1, j))
        if j + 1 < matrix_size:
            neighbors.append((i, j + 1))
        if i + 1 < matrix_size:
            neighbors.append((i + 1, j))
        return neighbors

matrix = [[0, 3, 2, "start"], [0, 0, 0, 0], [0, 4, 6, 8], [7, 9, 0, 10]]
hillcilmer = hillCimber(matrix)
def disply_path(path, matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if (i,j) in path:
                print(f"\033[92m{matrix[i][j]}\033[0m", end=" ")
            else:
                print(f"{matrix[i][j]}", end=" ")
        print()

def Test1():
    print("Test get_start")
    start = hillcilmer.get_start()
    if start == (0,3):
        print("Correct Start Returned")
    else:
        print("Incorrect Start Returned")

def Test2():
    print("Test get_neighbors 2 neighbors")
    start = (0,3)
    neighbors = hillcilmer.get_neighbors(start)
    if neighbors == [(0, 2), (1, 3)]:
        print("Correct Neighbors Returned")
    else:
        print("Incorrect Neighbors Returned")

def Test3():
    print("Test get_neighbors 4 neighbors")
    start = (1, 1)
    neighbors = hillcilmer.get_neighbors(start)
    if neighbors == [(1, 0), (0, 1), (1, 2), (2, 1)]:
        print("Correct Neighbors Returned")
    else:
        print("Incorrect Neighbors Returned")

Test1()
Test2()
Test3()

# Hill Cimber
print("No Simulated annealing")
path, end = hillcilmer.two_dimensional_hill_climbing()
print(f"The path taken by the vaccum is: {path} and ends at {end}")
disply_path(path, matrix)
print()

print("Simulated annealing")
path, end = hillcilmer.two_dimensional_hill_climbing(sim_ann=True)
print(f"The path taken by the vaccum is: {path} and ends at {end}")
disply_path(path, matrix)
print()