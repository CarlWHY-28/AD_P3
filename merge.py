import os
import csv
import time

import matplotlib.pyplot as plt
from scipy.stats import wilcoxon


def read_map(file_path):
    with open(file_path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        matrix = []
        for row in reader:
            matrix.append(list(map(float, row)))  # int will raise error, weird:(
    return matrix


def plot_times(t_dict, f_names):
    print(len(f_names))
    a_name = list(t_dict.keys())
    n = len(a_name)
    n_g = len(f_names)
    x = range(1, n_g + 1)
    print(x)

    plt.figure(figsize=(12, 8))
    for a in a_name:
        plt.plot(x, t_dict[a], marker='o', label=a)

    plt.xlabel('Graph')
    plt.ylabel('Time (s)')
    plt.title('Algorithm Times')
    plt.xticks(x, [f'{f_name}' for f_name in f_names], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_weights(w_dict, f_names):
    print(len(f_names))

    best = [float('inf') for i in range(len(f_names))]
    for i in range(len(f_names)):
        for key in w_dict:
            if w_dict[key][i] < best[i]:
                best[i] = w_dict[key][i]
    print(best)

    a_name = list(w_dict.keys())
    n = len(a_name)
    n_g = len(f_names)
    x = range(1, n_g + 1)

    plt.figure(figsize=(12, 8))
    for a in a_name:
        plt.plot(x, [100 * best[i] / w for i, w in enumerate(w_dict[a])], marker='o', label=a)

    plt.xlabel('Graph')
    plt.ylabel('Weight (%)')
    plt.title('Algorithm Weights Score')
    plt.xticks(x, [f'{f_name}' for f_name in f_names], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()


# A
def nearest_neighbor(adj_matrix, start=0):
    n = len(adj_matrix)
    visited = [False] * n
    tour = [start]
    total_weight = 0
    cur = start
    visited[cur] = True

    for ff in range(n - 1):
        nearest = None
        min_dist = float('inf')
        for neighbor in range(n):
            if not visited[neighbor] and adj_matrix[cur][neighbor] < min_dist and adj_matrix[cur][neighbor] != 0:
                min_dist = adj_matrix[cur][neighbor]
                nearest = neighbor
            else:
                continue
        tour.append(nearest)
        total_weight += min_dist
        cur = nearest
        visited[cur] = True

    # if adj_matrix[cur][start] == 0:
    #     print('111111111')
    tour.append(start)
    total_weight += adj_matrix[cur][start]
    return tour, total_weight


def g_t(tour):
    str = ''
    for i in range(len(tour) - 1):
        str += f'{tour[i]}->'
    str += f'{tour[0]}'
    return str


# B
def nearest_insertion(adj_matrix):
    n = len(adj_matrix)
    min_weight = float('inf')
    start_link = (0, 0)
    for i in range(n):
        for j in range(i + 1, n):
            if adj_matrix[i][j] < min_weight and adj_matrix[i][j] != 0:
                min_weight = adj_matrix[i][j]
                start_link = (i, j)  # shortest edge
    # print(f"Start Link: {start_link}")

    tour = list(start_link) + [start_link[0]]
    total_weight = adj_matrix[start_link[0]][start_link[1]] * 2
    visited = set(tour)

    while len(visited) < n:
        # Find
        nearest_node = None
        min_dist = float('inf')
        for node in range(n):
            if node not in visited:
                for tour_node in tour[:-1]:
                    if adj_matrix[node][tour_node] < min_dist and adj_matrix[node][tour_node] != 0:
                        min_dist = adj_matrix[node][tour_node]
                        nearest_node = node
        # print(f"Nearest Node: {nearest_node}")

        # Insert
        best_w = float('inf')
        best_p = -1
        for i in range(len(tour) - 1):
            a = tour[i]
            b = tour[i + 1]
            if adj_matrix[a][nearest_node] != 0 and adj_matrix[nearest_node][b] != 0:
                m_w = adj_matrix[a][nearest_node] + adj_matrix[nearest_node][b] - adj_matrix[a][b]
                if m_w < best_w:
                    best_w = m_w
                    best_p = i + 1

        tour.insert(best_p, nearest_node)
        total_weight += best_w
        visited.add(nearest_node)
    return tour, total_weight


# C
def dynamic_programming_tsp(adj_matrix, start=0):
    n = len(adj_matrix)
    memo = {}

    def visit(mask, cur):
        if (mask, cur) in memo:
            return memo[(mask, cur)]
        if mask == (1 << n) - 1:
            return adj_matrix[cur][start], [start]

        min_cost = float('inf')
        path = []
        for neighbor in range(n):
            if adj_matrix[cur][neighbor] != 0 and not (mask & (1 << neighbor)):  # not visited and not self
                # print(mask)
                new_cost, new_path = visit(mask | (1 << neighbor), neighbor)
                total_cost = adj_matrix[cur][neighbor] + new_cost
                if total_cost < min_cost:  # find the min cost
                    min_cost = total_cost
                    path = [neighbor] + new_path
        memo[(mask, cur)] = (min_cost, path)
        return memo[(mask, cur)]

    total_cost, path = visit(1 << start, start)
    # Reconstruct
    tour = [start] + path
    return tour, total_cost


# D. Branch-and-Bound
def branch_and_bound_tsp(adj_matrix, start=0):
    n = len(adj_matrix)
    best_cost = float('inf')
    best_tour = []

    def length(tour):
        total = 0
        for i in range(len(tour) - 1):
            total += adj_matrix[tour[i]][tour[i + 1]]
        total += adj_matrix[tour[-1]][tour[0]]
        return total

    def bound(cur, visited, cur_cost):
        lb = cur_cost
        min_out = float('inf')
        for i in range(n):
            if i not in visited and adj_matrix[cur][i] < min_out and adj_matrix[cur][i] != 0:
                min_out = adj_matrix[cur][i]
        if min_out != float('inf'):
            lb += min_out
        else:
            lb += 0
        for i in range(n):
            if i not in visited and i != start:
                min_in = float('inf')
                for j in range(n):
                    if j != i and adj_matrix[j][i] < min_in and adj_matrix[j][i] != 0:
                        min_in = adj_matrix[j][i]
                if min_in != float('inf'):
                    lb += min_in
                else:
                    lb += 0

        return lb

    def backtrack(cur, visited, cur_cost, tour):
        nonlocal best_cost, best_tour
        if len(visited) == n:
            if adj_matrix[cur][start] != 0:
                total_cost = cur_cost + adj_matrix[cur][start]
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_tour = tour + [start]
            return
        lb = bound(cur, visited, cur_cost)
        if lb >= best_cost:
            return

        for next_node in range(n):
            if next_node not in visited and adj_matrix[cur][next_node] != 0:
                visited_next = visited.copy()
                visited_next.add(next_node)
                next_cost = cur_cost + adj_matrix[cur][next_node]
                backtrack(next_node, visited_next, next_cost, tour + [next_node])

    backtrack(start, {start}, 0, [start])
    return best_tour, best_cost


def print_result(tour, cost, name):
    print(f"{name}")
    print(f"Tour: {tour}")
    print(f"Total Weight: {cost}")


def main(directory_path):
    output_file = open('output.csv', 'w', newline='')
    output_file.write('Algorithm,File,Tour: Sequence of nodes,Total weight\n')
    files = [f for f in os.listdir(directory_path) if f.endswith('.csv') and not f.startswith('.')]
    files = sorted(files, key=lambda x: int(x.split('n')[0]))

    times = {
        'Nearest Neighbor': [],
        'Nearest Insertion': [],
        'Dynamic Programming': [],
        'Branch-and-Bound': []
    }

    output_dict = {
        'Nearest Neighbor': [],
        'Nearest Insertion': [],
        'Dynamic Programming': [],
        'Branch-and-Bound': []
    }

    weights = {
        'Nearest Neighbor': [],
        'Nearest Insertion': [],
        'Dynamic Programming': [],
        'Branch-and-Bound': []
    }

    for idx, file in enumerate(files):
        file_path = os.path.join(directory_path, file)
        adj_matrix = read_map(file_path)

        # print(adj_matrix)
        n = len(adj_matrix)
        print("\n" + file)

        # A. Nearest Neighbor
        t0 = time.time()
        tour_nn, cost_nn = nearest_neighbor(adj_matrix, start=0)
        t1 = time.time()
        times['Nearest Neighbor'].append(t1 - t0)
        weights['Nearest Neighbor'].append(cost_nn)
        tour = g_t(tour_nn)
        print_result(tour, cost_nn, "A. Nearest Neighbor")
        output_dict['Nearest Neighbor'].append((file, tour, cost_nn))

        # B. Nearest Insertion
        t0 = time.time()
        tour_ni, cost_ni = nearest_insertion(adj_matrix)
        t1 = time.time()
        times['Nearest Insertion'].append(t1 - t0)
        weights['Nearest Insertion'].append(cost_ni)
        tour = g_t(tour_ni)
        print_result(tour, cost_ni, "B. Nearest Insertion")
        output_dict['Nearest Insertion'].append((file, tour, cost_ni))
        # output_file.write(f"Nearest Insertion,{file},{tour},{cost_ni}\n")

        # C. Dynamic Programming
        t0 = time.time()
        tour_dc, cost_dc = dynamic_programming_tsp(adj_matrix, start=0)
        t1 = time.time()
        times['Dynamic Programming'].append(t1 - t0)
        weights['Dynamic Programming'].append(cost_dc)
        tour = g_t(tour_dc)
        print_result(tour, cost_dc, "C. Dynamic Programming")
        output_dict['Dynamic Programming'].append((file, tour, cost_dc))

        # D. Branch-and-Bound
        t0 = time.time()
        tour_bb, cost_bb = branch_and_bound_tsp(adj_matrix, start=0)
        t1 = time.time()
        times['Branch-and-Bound'].append(t1 - t0)
        weights['Branch-and-Bound'].append(cost_bb)
        tour = g_t(tour_bb)
        print_result(tour, cost_bb, "D. Branch-and-Bound")
        output_dict['Branch-and-Bound'].append((file, tour, cost_bb))

    for key in output_dict:
        for ff in output_dict[key]:
            output_file.write(f"{key},{ff[0]},{ff[1]},{ff[2]}\n")

    plot_times(times, files)
    plot_weights(weights, files)
    output_file.close()


if __name__ == "__main__":
    main('/Users/whyyy/Downloads/Project3-Input-Files')
