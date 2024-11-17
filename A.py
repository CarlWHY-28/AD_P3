import os
import csv

def nn(adj_matrix, start=0):
    n = len(adj_matrix)
    visited = [False] * n
    tour = [start]
    total_weight = 0
    cur = start
    visited[cur] = True

    for ff in range(n - 1):# no start
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
        tour, cost = nn(adj_matrix, start=0)

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
