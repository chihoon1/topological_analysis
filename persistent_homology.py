'''
Implementation of Persistent homology

Persistent homology comprises of all homology Hk(Xr) for all k >= 0
dimension of k-th homology Hk(Xr) indicates the number of independent k-dimensional holes in the topological space X
'''
import matplotlib.pyplot as plt
import seaborn as sns
from persistent_vector import *
from homology import compute_Homology_k
from sklearn.datasets import make_circles


'''
Incorporating all functions from all the other scripts,
compute the dimension of persistent homology of the given metric space (X,d).
However, k must be always given as an input value to fix the persistent homology at k.

In this step, if the input value r is given as a specified integer,
then the function named metric_space_to_PHk_Xr will basically compute the dimension of k-th homology of VR(X[r]) at the given r.
𝐼𝑓 𝑛𝑜 𝑟 𝑖𝑠 𝑔𝑖𝑣𝑒𝑛 𝑎𝑠 𝑖𝑛𝑝𝑢𝑡 𝑣𝑎𝑙𝑢𝑒, 𝑡ℎ𝑒𝑛 𝑡ℎ𝑒 𝑓𝑢𝑛𝑐𝑡𝑖𝑜𝑛 𝑤𝑖𝑙𝑙 𝑐𝑜𝑚𝑝𝑢𝑡𝑒 𝑡ℎ𝑒 𝑑𝑖𝑚𝑒𝑛𝑠𝑖𝑜𝑛 𝑜𝑓 𝑃𝐻𝑘(𝑋)𝑟 𝑓𝑜𝑟 𝑎𝑙𝑙 𝑟.
'''
def metric_space_to_PHk_Xr(X, k, r=None, d=euclidean_distance):
    # Compute dimension of k-th persistent homology based on Vietoris Rips Complex
    # param: X is a data set
    # param: d is a distance metric function(taking two data points as parameters). default=euclidean
    # param: r(integer) >= 0 and criteria for forming a simplex. Default: finite set of all r
    # param: k(int). Highest dimensional simplex will be k simplex for the output. must be given
    # ex) k=(ex. n-1 simplex (n=# of data points))
    # return a tuple containing r(int or list if r=None) and dimension of PHk(X)r(int or list if r=None).
    if k is None: k = len(X) - 1
    if r is not None:
        # get simplices of VR(x[r]) for fixed r
        adjacency_matrix = get_adjacency_matrix(X, d=d)
        simplicial_comp = metric_space_to_VRComplex(adjacency_matrix, r=r)
        # return the k-th homology of VR(X[r])
        return r, compute_Homology_k(simplicial_comp, k)
    else:
        # get persistent vector space of X for all finite number of r
        finite_set_r, pvs = metric_space_to_PersVecSpace(X, d=d)
        # compute the k-th homology of VR(X[r]) for all r
        dim_pers_homology = []
        for vietoris_rips in pvs:
            dim_pers_homology.append(compute_Homology_k(vietoris_rips, k))
        return finite_set_r, dim_pers_homology


if __name__ == '__main__':
    # test the persistent homolgy for data analysis
    # form a data (unit circle in R2 but with some marginal errors around the circle)
    data = make_circles(n_samples=17, shuffle=True, noise=None, random_state=0, factor=0.99)[0]
    for i in range(len(data)):
        error = np.random.uniform(low=-0.1, high=0.1, size=2)
        data[i] += error

    # plot the dataset
    plt.figure(figsize=(9, 9))
    sns.scatterplot(x=data[:, 0], y=data[:, 1])

    # Get all possible finite number of r
    adjacency_matrix = get_adjacency_matrix(data)
    finite_set_of_r = sorted(set(adjacency_matrix.flatten()))

    # three examples of k-th PH(X)r with fixed values of r
    n = len(finite_set_of_r) - 1
    # 0-th persistent homology for fixed r
    print("0-th persistent homology")
    r, phk_X = metric_space_to_PHk_Xr(data, 0, r=finite_set_of_r[np.random.randint(0, n)])
    print(f"r value:\n{r}")
    print(f"dimension of 0-th persistent homology of r={r}:\n{phk_X}")
    r, phk_X = metric_space_to_PHk_Xr(data, 0, r=finite_set_of_r[np.random.randint(0, n)])
    print(f"r value:\n{r}")
    print(f"dimension of 0-th persistent homology of r={r}:\n{phk_X}")
    r, phk_X = metric_space_to_PHk_Xr(data, 0, r=finite_set_of_r[np.random.randint(0, n)])
    print(f"r value:\n{r}")
    print(f"dimension of 0-th persistent homology of r={r}:\n{phk_X}\n")

    # 1-th persistent homology for all finite number of r
    print("1-th persistent homology")
    r, phk_X = metric_space_to_PHk_Xr(data, 1, r=finite_set_of_r[np.random.randint(0, n)])
    print(f"r value:\n{r}")
    print(f"dimension of 1-th persistent homology of r={r}:\n{phk_X}")
    r, phk_X = metric_space_to_PHk_Xr(data, 1, r=finite_set_of_r[np.random.randint(0, n)])
    print(f"r value:\n{r}")
    print(f"dimension of 1-th persistent homology of r={r}:\n{phk_X}")
    r, phk_X = metric_space_to_PHk_Xr(data, 1, r=finite_set_of_r[np.random.randint(0, n)])
    print(f"r value:\n{r}")
    print(f"dimension of 1-th persistent homology of r={r}:\n{phk_X}")

    # now applying the algorithms to our example data set for all r
    # 0-th persistent homology for all finite number of r
    all_r, phk_X = metric_space_to_PHk_Xr(data, 0)
    print(f"all possible r values:\n{all_r}")
    print(f"dimension of 0-th persistent homology in the ascending order of r:\n{phk_X}")
    # 1-th persistent homology for all finite number of r
    all_r, phk_X = metric_space_to_PHk_Xr(data, 1)
    print(f"all possible r values:\n{all_r}")
    print(f"dimension of 1-th persistent homology in the ascending order of r:\n{phk_X}")

