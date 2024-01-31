This project utilize Topological methods to analyze data.

Here, I implemented persistent homology to analyze the shape of data distribution and its topological features.


About Persistent Homology:
Persistent homology comprises of all homology Hk(Xr) for all k >= 0
dimension of k-th homology Hk(Xr) indicates the number of independent k-dimensional holes in the topological space X (simplicial complex)

k-dimension in simplicial complex
1. 0-dimension: a point
2. 1-dimension: a line segment
3. 2-dimension: a triangle
4. 3-dimension: a tetrahedron
5. 4-dimension: 5-cell

Here are the steps I took to compute the persistent homology:

1. Turn finite metric space data into abstract complex (Viterois Rips Complex)
2. Combine all VR complex to make a finite persistent vector space
3. Get boundary matrix of chain complex Ck and Ck+1 and do row_reduction
4. Compute the dimension of ker and img of boundary maps dk and dk+1 to get dim H_k(X)_r for fixed r
5. Compute the dimension of H_k(X)_r for all r (if needed)


How to use the program:

1. git clone https://github.com/chihoon1/topological_analysis.git  # install this git repo to local computer
2. python -m venv venv  # set up virtual environment for python project
3. source venv/bin/active  # activate virtual environment
4. pip install -r requirements.txt  # install all the library packages needed for persistent homology algorithm
5. invoke function named metric_space_to_PHk_Xr to analyze data with persistent homology


Algorithm is tested in persistent_homology_analysis.ipynb or in the main function of persistent_homology.py
