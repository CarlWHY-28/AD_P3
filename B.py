import os
import csv


def ni(adj_matrix):
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


def main(fpath):
    output_file = open('output.csv', 'w', newline='')
    output_file.write('Algorithm,File,Tour: Sequence of nodes,Total weight\n')
    files = [f for f in os.listdir(fpath) if f.endswith('.csv') and not f.startswith('.')]
    files = sorted(files, key=lambda x: int(x.split('n')[0]))

    for idx, file in enumerate(files):
        file_path = os.path.join(fpath, file)
        with open(file_path, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            adj_matrix = []
            for row in reader:
                adj_matrix.append(list(map(float, row)))  # int will raise error, weird:(

        n = len(adj_matrix)
        print("\n" + file)
        tour, cost = ni(adj_matrix)

        str = ''
        for i in range(len(tour) - 1):
            str += f'{tour[i]}->'
        str += f'{tour[0]}'

        print(f"Tour: {str}")
        print(f"Total Weight: {cost}")
        output_file.write(f"Nearest Neighbor,{file},{str},{cost}\n")
    output_file.close()


if __name__ == "__main__":
    main('/Users/whyyy/Downloads/Project3-Input-Files')
