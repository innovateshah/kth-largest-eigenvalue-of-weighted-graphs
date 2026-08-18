# kth-largest-eigenvalue-of-weighted-graphs

Python code for searching for graphs with large **k-th largest adjacency-matrix eigenvalues**.

The project generates random symmetric weighted matrices, optimizes them using their k-th eigenvector, converts them to binary adjacency matrices, and keeps non-isomorphic graphs achieving the best eigenvalue found.

## Requirements

```bash
pip install numpy pandas scipy networkx
```

Dependencies:
* NumPy
* Pandas
* SciPy
* NetworkX
---

## Function in `kthsup.py`

### `randsym(n)`

Generates a random symmetric `n × n` matrix with entries in `[0,1]` and zero diagonal.

### `H_ab(a, b)`

Constructs the adjacency matrix for the graph with maximal 3rd largest eigenvalue as described by

```text
a, b, a, b, a, b
```

Each group is complete, and adjacent groups are completely connected.

Returns the adjacency matrix.

### `kth(A, k)`

Returns the k-th largest eigenvalue of a symmetric matrix and its corresponding eigenvector using `scipy.linalg.eigh`.

```python
val, vec = kth(A, k)
```

### `optimize(A, k, epochs=10**5, rate=1e-3)`

Iteratively increases the k-th eigenvalue using the update

```python
P = rate * np.outer(vec, vec)
A = np.clip(A + P, 0, 1)
```

The diagonal is kept at zero.

Returns:

```python
val, weighted_matrix
```

### `isomorphic(A, B)`

Checks whether two adjacency matrices represent isomorphic graphs using NetworkX.

### `search_optimal(...)`

Performs a randomized search:

1. Generate a random symmetric matrix.
2. Optimize it.
3. Threshold entries at `0.5` to produce a binary adjacency matrix.
4. Calculate its k-th largest eigenvalue.
5. Keep only candidates achieving the current maximum.
6. Remove isomorphic duplicates.
7. Save results to JSON.

Results are stored as:

```text
kth{k},{n}.json
```

with columns:

| Column | Description               |
| ------ | ------------------------- |
| `kth`  | k-th largest eigenvalue   |
| `W`    | Optimized weighted matrix |
| `A`    | Binary adjacency matrix   |

### `parse_adjacency_matrices(filename)`

Reads adjacency matrices from a text file formatted as:

```text
graph_id: matrix
```

Returns a dictionary mapping graph IDs to NumPy arrays.

### `auto_optimize(A, k, max_epochs=10**6, rate=1e-3, tol=0.25)`

Optimizes a matrix until it is sufficiently close to its binary thresholded version or `max_epochs` is reached.

The stopping condition is based on:

```python
np.linalg.norm(A - flattened) <= tol
```

### `auto_search_optimal(...)`

Search variant intended to use `auto_optimize` while applying the same maximum-eigenvalue and graph-isomorphism filtering as `search_optimal`.

---

## Driver Script

The second file loads a saved result for a specified `n` and `k`.

```python
n = 3 * 5 + 1
k = 3

path = "kth" + str(k) + "," + str(n) + ".json"
df = pd.read_json(path)
```

The weighted and binary adjacency matrices can be accessed with:

```python
W = np.asarray(df["W"].iloc[0])
A = np.asarray(df["A"].iloc[0])
```

A new search can be run with:

```python
df = sup.search_optimal(
    n,
    k,
    epochs1=20,
    epochs2=10**6,
    rate=2e-3,
    min_val=0
)
```

---

## Mathematical Method

For a symmetric weighted matrix `A`, let `v_k` be the eigenvector corresponding to its k-th largest eigenvalue.

Each optimization step essentially applies:

$$
A \leftarrow A + \eta v_kv_k^T
$$

where `η` is the learning rate and A is reformed to be a weighted graph (0's on the diagonal and weighted on [0,1]). This process stops when either the number of iterations exceeds epochs, or is "close" to an unweighted adjacency matrix (created by flattening the weights of the weighted matrix to 0 or 1). The k-th largest eigenvalue of the flattened matrix graph determines its score. The search retains non-isomorphic graphs tied for the best score found.

##Results

Here are the results found for n=7,10,13,16.
n=7: $\lambda_k=1.24697960372=2\cos(2\pi/7)$
Unique up to isomorphism.
The $C_7$ graph (cyclic).

n=10: $\lambda_k=2.2360679774997894$
Unique up to isomorphism.
Highly symmetric graph
Adjacency matrix:
$$
[[0 0 0 0 0 1 1 1 0 1]
 [0 0 0 1 1 0 0 1 1 0]
 [0 0 0 0 1 1 1 0 1 0]
 [0 1 0 0 1 0 0 1 0 1]
 [0 1 1 1 0 0 0 0 1 0]
 [1 0 1 0 0 0 1 0 0 1]
 [1 0 1 0 0 1 0 0 1 0]
 [1 1 0 1 0 0 0 0 0 1]
 [0 1 1 0 1 0 1 0 0 0]
 [1 0 0 1 0 1 0 1 0 0]]
$$

n=13: $\lambda_k=3.162522070908035$
Unique up to isomorphism.
Adjacency matrix:
$$
[[0 1 1 0 0 1 0 0 1 0 0 0 1]
 [1 0 1 0 0 0 0 0 1 0 1 0 1]
 [1 1 0 0 1 0 0 1 0 0 1 0 0]
 [0 0 0 0 0 1 1 0 0 1 0 1 0]
 [0 0 1 0 0 0 0 1 0 0 1 1 0]
 [1 0 0 1 0 0 1 0 1 1 0 0 1]
 [0 0 0 1 0 1 0 0 0 1 0 1 0]
 [0 0 1 0 1 0 0 0 0 0 1 1 0]
 [1 1 0 0 0 1 0 0 0 0 0 0 1]
 [0 0 0 1 0 1 1 0 0 0 0 1 0]
 [0 1 1 0 1 0 0 1 0 0 0 1 0]
 [0 0 0 1 1 0 1 1 0 1 1 0 0]
 [1 1 0 0 0 1 0 0 1 0 0 0 0]]
$$


n=16: $\lambda_k=4.162277660168379$
NOT unique up to isomorphism.
Adjacency matrices:
$$
[[0 1 0 0 1 0 0 1 0 1 0 0 1 0 1 0]
 [1 0 1 0 0 0 0 1 0 1 0 1 1 0 0 1]
 [0 1 0 0 0 0 0 0 1 1 1 1 0 0 0 1]
 [0 0 0 0 1 1 1 0 0 0 1 0 0 1 1 0]
 [1 0 0 1 0 1 1 1 0 0 0 0 1 0 1 0]
 [0 0 0 1 1 0 1 0 0 0 1 0 0 1 1 0]
 [0 0 0 1 1 1 0 0 1 0 1 0 0 1 1 0]
 [1 1 0 0 1 0 0 0 0 1 0 0 1 0 1 0]
 [0 0 1 0 0 0 1 0 0 0 1 1 0 1 0 1]
 [1 1 1 0 0 0 0 1 0 0 0 1 1 0 0 1]
 [0 0 1 1 0 1 1 0 1 0 0 1 0 1 0 1]
 [0 1 1 0 0 0 0 0 1 1 1 0 0 1 0 1]
 [1 1 0 0 1 0 0 1 0 1 0 0 0 0 1 0]
 [0 0 0 1 0 1 1 0 1 0 1 1 0 0 0 0]
 [1 0 0 1 1 1 1 1 0 0 0 0 1 0 0 0]
 [0 1 1 0 0 0 0 0 1 1 1 1 0 0 0 0]]

[[0 0 1 1 1 0 0 0 1 1 0 1 0 1 0 0]
 [0 0 0 1 0 1 0 1 0 1 0 1 1 0 1 0]
 [1 0 0 0 1 0 1 0 1 1 1 0 0 1 0 1]
 [1 1 0 0 0 0 0 1 1 1 0 1 0 0 0 0]
 [1 0 1 0 0 0 1 0 1 1 1 0 0 1 0 1]
 [0 1 0 0 0 0 1 1 0 0 1 0 1 0 1 1]
 [0 0 1 0 1 1 0 0 0 0 1 0 1 1 1 1]
 [0 1 0 1 0 1 0 0 0 1 0 1 1 0 1 0]
 [1 0 1 1 1 0 0 0 0 1 0 1 0 1 0 0]
 [1 1 1 1 1 0 0 1 1 0 0 1 0 0 0 0]
 [0 0 1 0 1 1 1 0 0 0 0 0 1 1 1 1]
 [1 1 0 1 0 0 0 1 1 1 0 0 1 0 1 0]
 [0 1 0 0 0 1 1 1 0 0 1 1 0 0 1 1]
 [1 0 1 0 1 0 1 0 1 0 1 0 0 0 0 1]
 [0 1 0 0 0 1 1 1 0 0 1 1 1 0 0 1]
 [0 0 1 0 1 1 1 0 0 0 1 0 1 1 1 0]]
$$


## Miscellaneous Notes

I do not have complete confidence that this algorithm actually find the weighted graph with the largest kth eigenvalue. All largely reducing to the matrix we apply is different from the projection matrix. I suspect clipping to be largest source however

When attempting to run the program, expect more than 60 seconds to run. n=16 case took ~60 seconds per graph optimization. When running, auto_search is recommended.

