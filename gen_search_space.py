import pickle

search_space = []
for i in range(2**18):
    ii = bin(i)[2:]  # str
    arch = [int(j) for j in '0' * (19-len(ii)) + ii]
    search_space.append(arch)

with open('search_space', 'wb') as file:
    pickle.dump(search_space, file)
