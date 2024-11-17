import os
import csv



def read_map(file_path):
    with open(file_path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        matrix = []
        for row in reader:
            matrix.append(list(map(float, row)))  # int will raise error, weird:(
    return matrix


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



def print_result(tour, cost, name):
    print(f"{name}")
    print(f"Tour: {tour}")
    print(f"Total Weight: {cost}")


def main(directory_path):
    output_file = open('output.csv', 'w', newline='')
    output_file.write('Algorithm,File,Tour: Sequence of nodes,Total weight\n')
    files = [f for f in os.listdir(directory_path) if f.endswith('.csv') and not f.startswith('.')]
    files = sorted(files, key=lambda x: int(x.split('n')[0]))


    for idx, file in enumerate(files):
        file_path = os.path.join(directory_path, file)
        adj_matrix = read_map(file_path)

        # print(adj_matrix)
        n = len(adj_matrix)
        print("\n" + file)

        # A. Nearest Neighbor
        tour_nn, cost_nn = nearest_neighbor(adj_matrix, start=0)
        tour = g_t(tour_nn)
        print_result(tour, cost_nn, "A. Nearest Neighbor")
        output_file.write(f"Nearest Neighbor,{file},{tour},{cost_nn}\n")

        # B. Nearest Insertion
        tour_ni, cost_ni = nearest_insertion(adj_matrix)
        tour = g_t(tour_ni)
        print_result(tour, cost_ni, "B. Nearest Insertion")
        output_file.write(f"Nearest Insertion,{file},{tour},{cost_ni}\n")

    output_file.close()


if __name__ == "__main__":
    main('/Users/whyyy/Downloads/Project3-Input-Files')
