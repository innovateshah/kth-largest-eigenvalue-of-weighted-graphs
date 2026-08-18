import numpy as np
import pandas as pd
from scipy.linalg import eigh
import networkx as nx
import os
import time

def randsym(n):
    M = np.random.rand(n, n)
    M = np.triu(M, 1)
    return M + M.T


def H_ab(a, b):
    """ Construct the adjacency matrix H_{a,b}. """

    if a < 1 or b < 1:
    	raise ValueError("a and b must be positive integers")

    # Sizes of the six groups around the hexagon
    sizes = [a, b, a, b, a, b]

    # Total number of vertices
    n = sum(sizes)

    H = np.zeros((n, n), dtype=int)

    # Starting index of each group
    starts = np.cumsum([0] + sizes[:-1])

    # ---------------------------------------------------------
    # 1. Make each group a complete graph
    # ---------------------------------------------------------
    for start, size in zip(starts, sizes):
        end = start + size

        # Complete graph K_size, with zero diagonal
        H[start:end, start:end] = 1
        np.fill_diagonal(H[start:end, start:end], 0)

    # ---------------------------------------------------------
    # 2. Complete bipartite connections between adjacent groups in the hexagon
    # ---------------------------------------------------------
    for i in range(6):
        j = (i + 1) % 6  # next group, wrapping around

        i_start, i_end = starts[i], starts[i] + sizes[i]
        j_start, j_end = starts[j], starts[j] + sizes[j]

        # Complete bipartite connection
        H[i_start:i_end, j_start:j_end] = 1
        H[j_start:j_end, i_start:i_end] = 1

    return H

def kth(A,k):
	n,_ = A.shape
	return eigh(A,subset_by_index=[n-k,n-k])

def optimize(A,k,epochs=10**5, rate=1e-3):
	bad_count = 0
	bad_inarow = 0
	max_inarow = 0

	n, _ = A.shape
	val, vec = kth(A,k)
	for _ in range(epochs):
		P = rate*np.outer(vec,vec)
		np.fill_diagonal(P,0)
		A = np.clip(A+P,0,1)
		val, vec = kth(A,k)
		#t_val,t_vec = kth(New,k)
		"""
		if(t_val > val):
			val = t_val
			vec = t_vec
			A = New
			if(max_inarow < bad_inarow):
				max_inarow = bad_inarow
			bad_inarow=0
		else:
			bad_count+=1
			bad_inarow+=1
			#print("Proj was worse " +str(bad_count))
			A = np.clip(A-P,0,1)
			val, vec = kth(A,k)
		"""
	print(f"total bad: {bad_count}, in a row: {max(max_inarow,bad_inarow)}")
	return val, A

def isomorphic(A,B):
	G1=nx.from_numpy_array(A)
	G2=nx.from_numpy_array(B)
	return nx.is_isomorphic(G1,G2)

def search_optimal(n, k, epochs1, epochs2, rate):
    """
    Generate random symmetric matrices, optimize them, flatten them,
    and save only non-isomorphic graphs meeting the kth-eigenvalue
    criterion.

    Parameters
    ----------
    n : int
        Matrix size.
    k : int
        Eigenvalue index.
    epochs1 : int
        Number of random matrices to generate.
    epochs2 : int
        Number of optimization iterations per matrix.
    rate : float
        Optimization step size.
    min_val : float
        Minimum allowed kth eigenvalue for a flattened matrix.

    The DataFrame is maintained so that all entries have the same
    kth eigenvalue. Whenever a new maximum kth eigenvalue is found,
    the entire DataFrame is cleared and only the new maximum is kept.
    """
    tol = rate**2
    path = "kth" + str(k) + "," + str(n) + ".json"

    if os.path.exists(path):
        df = pd.read_json(path)
    else:
        df = pd.DataFrame(columns=["kth", "W", "A"])

    # Current maximum kth eigenvalue
    if len(df) == 0:
        max_kth_eigval = -np.inf
    else:
        max_kth_eigval = df["kth"].max()

    # Reconstruct graphs from existing flattened matrices
    existing_graphs = [
        nx.from_numpy_array(np.array(A))
        for A in df["A"]
    ]

    for epoch in range(epochs1):

        # Generate random symmetric weighted matrix
        A = randsym(n)

        # Optimize
        val, weighted = optimize(
            A,
            k,
            epochs=epochs2,
            rate=rate
        )

        # Flatten optimized matrix
        flattened = np.where(weighted > 0.5, 1, 0)

        # kth eigenvalue of flattened matrix
        flat_val = float(kth(flattened, k)[0][0])

        # ---------------------------------------------------------
        # Minimum eigenvalue test
        # ---------------------------------------------------------
        if flat_val < min_val-tol:
            print(
                f"Epoch {epoch + 1}/{epochs1}: "
                f"{flat_val} < min_val {min_val} -- dropped"
            )
            continue

        # ---------------------------------------------------------
        # New maximum: clear the entire DataFrame
        # ---------------------------------------------------------
        if flat_val > max_kth_eigval+tol:
            max_kth_eigval = flat_val

            df = pd.DataFrame(
                [{
                    "kth": flat_val,
                    "W": weighted.tolist(),
                    "A": flattened.tolist()
                }]
            )

            # The old graphs are no longer relevant
            existing_graphs = [
                nx.from_numpy_array(flattened)
            ]

            # Save immediately
            df.to_json(path)

            print(
                f"Epoch {epoch + 1}/{epochs1}: "
                f"NEW MAXIMUM -- kth eigenvalue = {flat_val}; "
                f"DataFrame cleared"
            )
            continue

        # ---------------------------------------------------------
        # Only consider matrices at the current maximum
        # ---------------------------------------------------------
        if flat_val < max_kth_eigval - tol:
            print(
                f"Epoch {epoch + 1}/{epochs1}: "
                f"{flat_val} < current maximum {max_kth_eigval} -- dropped"
            )
            continue

        # ---------------------------------------------------------
        # Isomorphism test
        # ---------------------------------------------------------
        G = nx.from_numpy_array(flattened)

        if any(nx.is_isomorphic(G, old_G) for old_G in existing_graphs):
            print(
                f"Epoch {epoch + 1}/{epochs1}: "
                f"isomorphic -- dropped"
            )
            continue

        # ---------------------------------------------------------
        # Add another matrix with the SAME maximum kth eigenvalue
        # ---------------------------------------------------------
        df.loc[len(df)] = [
            flat_val,
            weighted.tolist(),
            flattened.tolist()
        ]

        existing_graphs.append(G)

        # Save JSON
        df.to_json(path)

        print(
            f"Epoch {epoch + 1}/{epochs1}: "
            f"ADDED -- kth eigenvalue = {flat_val}"
        )

    return df

def parse_adjacency_matrices(filename):
    matrices = {}

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            # Split graph ID from adjacency matrix
            graph_id, matrix_str = line.split(":", 1)

            # Convert each binary string into a list of integers
            matrix = np.array([
                [int(x) for x in row]
                for row in matrix_str.split()
            ])

            matrices[int(graph_id)] = matrix

    return matrices


def auto_optimize(A, k, max_epochs=10**6,rate=1e-3, tol=0.25):
    n, _ = A.shape
    flattened = np.where(A > .5, 1, 0)
    val, vec = kth(A,k)
    count = 0
    precent = 0

    start = time.perf_counter()
    while np.linalg.norm(A-flattened)>tol and count < max_epochs:
        P = rate*np.outer(vec,vec)
        np.fill_diagonal(P,0)
        A = np.clip(A+P,0,1)
        val, vec = kth(A,k)
        flattened = np.where(A>.5,1,0)
        if(count%(max_epochs/10) == 0):
            print(f"Precent: {10*precent}%")
            precent+=1
        count+=1
    stop = time.perf_counter()
    if(count > max_epochs):
        print(f"Hit max_epochs, time: {stop-start}")
    else:
        print(f"Hit tolerence, precent: {count/max_epochs}, time: {stop-start}")
    return val, A

#Same as search optimal but uses auto optimize instead to have higher ensurance that matrix is optimal
def auto_search_optimal(n, k, min_val, epochs1=10, epochs2=10**7, rate=1e-3, norm_tol=.3):
    """
    Generate random symmetric matrices, optimize them, flatten them,
    and save only non-isomorphic graphs meeting the kth-eigenvalue
    criterion.

    Parameters
    ----------
    n : int
        Matrix size.
    k : int
        Eigenvalue index.
    epochs1 : int
        Number of random matrices to generate.
    epochs2 : int
        Number of optimization iterations per matrix.
    rate : float
        Optimization step size.
    min_val : float
        Minimum allowed kth eigenvalue for a flattened matrix.

    The DataFrame is maintained so that all entries have the same
    kth eigenvalue. Whenever a new maximum kth eigenvalue is found,
    the entire DataFrame is cleared and only the new maximum is kept.
    """
    tol = rate**2
    path = "kth" + str(k) + "," + str(n) + ".json"

    if os.path.exists(path):
        df = pd.read_json(path)
    else:
        df = pd.DataFrame(columns=["kth", "W", "A"])

    # Current maximum kth eigenvalue
    if len(df) == 0:
        max_kth_eigval = -np.inf
    else:
        max_kth_eigval = df["kth"].max()

    # Reconstruct graphs from existing flattened matrices
    existing_graphs = [
        nx.from_numpy_array(np.array(A))
        for A in df["A"]
    ]

    for epoch in range(epochs1):

        # Generate random symmetric weighted matrix
        A = randsym(n)

        # Optimize
        val, weighted = auto_optimize(
            A,
            k,
            max_epochs=epochs2,
            rate=rate,
            tol=norm_tol
        )

        # Flatten optimized matrix
        flattened = np.where(weighted > 0.5, 1, 0)

        # kth eigenvalue of flattened matrix
        flat_val = float(kth(flattened, k)[0][0])

        # ---------------------------------------------------------
        # Minimum eigenvalue test
        # ---------------------------------------------------------
        if flat_val < min_val-tol:
            print(
                f"Epoch {epoch + 1}/{epochs1}: "
                f"{flat_val} < min_val {min_val} -- dropped"
            )
            continue

        # ---------------------------------------------------------
        # New maximum: clear the entire DataFrame
        # ---------------------------------------------------------
        if flat_val > max_kth_eigval+tol:
            max_kth_eigval = flat_val

            df = pd.DataFrame(
                [{
                    "kth": flat_val,
                    "W": weighted.tolist(),
                    "A": flattened.tolist()
                }]
            )

            # The old graphs are no longer relevant
            existing_graphs = [
                nx.from_numpy_array(flattened)
            ]

            # Save immediately
            df.to_json(path)

            print(
                f"Epoch {epoch + 1}/{epochs1}: "
                f"NEW MAXIMUM -- kth eigenvalue = {flat_val}; "
                f"DataFrame cleared"
            )
            continue

        # ---------------------------------------------------------
        # Only consider matrices at the current maximum
        # ---------------------------------------------------------
        if flat_val < max_kth_eigval - tol:
            print(
                f"Epoch {epoch + 1}/{epochs1}: "
                f"{flat_val} < current maximum {max_kth_eigval} -- dropped"
            )
            continue

        # ---------------------------------------------------------
        # Isomorphism test
        # ---------------------------------------------------------
        G = nx.from_numpy_array(flattened)

        if any(nx.is_isomorphic(G, old_G) for old_G in existing_graphs):
            print(
                f"Epoch {epoch + 1}/{epochs1}: "
                f"isomorphic -- dropped"
            )
            continue

        # ---------------------------------------------------------
        # Add another matrix with the SAME maximum kth eigenvalue
        # ---------------------------------------------------------
        df.loc[len(df)] = [
            flat_val,
            weighted.tolist(),
            flattened.tolist()
        ]

        existing_graphs.append(G)

        # Save JSON
        df.to_json(path)

        print(
            f"Epoch {epoch + 1}/{epochs1}: "
            f"ADDED -- kth eigenvalue = {flat_val}"
        )

    return df