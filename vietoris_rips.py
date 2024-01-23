'''
Transforming metric space to Vietoris Rips Complex

Function find_big_simplex finds the highest dimension simplex that can be formed with two vertices vi and vj(for all j)
After obtaining highest dimension simplices using the function(find_big_simplex),
found all faces of those simplices using powerset in the function named metric_space_to_VRComplex.
Combining all the faces and highest dimension simplices will produce an abstract simplicial complex.
Here, abstract simplicial complex is Vietoris Rips Complex at one fixed r.
'''


from itertools import chain, combinations


# Time Complexity: O(n^2)
def find_big_simplex(adj_mat, r, vi, k):
    # find the highest dimension simplex can be formed with two vertices vi and vj(for all j)
    # param: adj_mat(np.2d array) is an adjacency matrix
    # param: r is the distance criteria
    # param: vi(int) is a index of vertex vi
    # param: k(int). Highest dimensional simplex will be k simplex for the output.
    # k=(ex. n-1 simplex (n=# of data points))
    # return high dimension simplices involving vi and vj(vj is a vertex, but vj != vi)
    # return type: a set of sets where the inner set represents verticies of a simplex
    simplexes_set = set()
    for col_ind in range(adj_mat.shape[1]):
        i = vi
        vert_walked = {i}
        simplex = {i}
        if adj_mat[i,col_ind] > r:
            continue
        vert_walked.add(col_ind)
        simplex.add(col_ind)
        i = col_ind
        while len(simplex) < k + 1 and i != None:
            for j in range(i, adj_mat.shape[1]):
                if (adj_mat[i,j] <= r) and (j not in vert_walked):
                    # check all edges between j(current index) and vl(belongs to current simplex vertices)
                    # <= r
                    switch = 1
                    for vl in simplex:
                        if adj_mat[j,vl] > r:
                            switch = 0
                            break
                    if switch:  # true if adding j form a higher dimension simplex. so, move to j-th row
                        vert_walked.add(j)
                        simplex.add(j)
                        i = j
                        break
            else:
                i = None
        simplexes_set.add(frozenset(simplex))
    return simplexes_set


def metric_space_to_VRComplex(adj_mat, r, k=None):
    # turn finite metric space data into abstract complex (Vitoris Rips Complex) with param r
    # param: adj_mat is an adjacency matrix (2d array)
    # param: r(integer)>=0 and criteria for forming a simplex
    # param: k(int). Highest dimensional simplex will be k simplex for the output.
    # k=(ex. n-1 simplex (n=# of data points)). default k=(ex. n-1 simplex (n=# of data points))
    # return VR complex(VR(X[r])) a set of sets where the inner set represents verticies of a simplex
    if k is None: k = len(adj_mat)-1
    simplicial_comp = set()
    is_everything_simplex = False  # True if n-1 simplex in VR(x[r]) where n=# of data points
    for i in range(adj_mat.shape[0]):
        s = find_big_simplex(adj_mat, r=r, vi=i, k=k)
        subsets = set()
        for elem in s:
            powerset = set(chain.from_iterable(combinations(elem, r) for r in range(len(elem)+1)))
            subsets = subsets.union(powerset)
            if len(elem) == adj_mat.shape[0]:
                # if n-1 simplex, then its powerset is all possible subsets of Vertex(X).
                # In this case, no need to compute powerset of other simplices
                is_everything_simplex = True
                break
        simplicial_comp = simplicial_comp.union(subsets)
        if is_everything_simplex:
            break
    simplicial_comp = simplicial_comp.difference({()})  # exclude empty set
    return simplicial_comp





