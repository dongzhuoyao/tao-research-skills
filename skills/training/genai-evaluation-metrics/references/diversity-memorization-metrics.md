# Diversity & Memorization Metrics Implementation

## PRDC: Precision, Recall, Density, Coverage

Manifold-based metrics using k-NN in feature space:

```python
class PRDC(Metric):
    def __init__(self, nearest_k=5):
    def compute(self):
        real_nn = compute_nearest_neighbour_distances(real_feats, self.nearest_k)
        fake_nn = compute_nearest_neighbour_distances(fake_feats, self.nearest_k)
        dist_rf = compute_pairwise_distance(real_feats, fake_feats)
        precision = (dist_rf < real_nn[:, None]).any(axis=0).mean()   # Fidelity
        recall    = (dist_rf < fake_nn[None, :]).any(axis=1).mean()   # Mode coverage
        density   = (dist_rf < real_nn[:, None]).sum(axis=0).mean() / self.nearest_k
        coverage  = (dist_rf.min(axis=1) < real_nn).mean()            # Support coverage
```

## AuthPct: Memorization Detection

Detects if generated samples are copies of training data:

```python
def compute_authpct(train_feat, gen_feat):
    real_dists = torch.cdist(train_feat, train_feat)
    real_dists.fill_diagonal_(float("inf"))
    gen_dists = torch.cdist(train_feat, gen_feat)
    # For each fake: is its nearest real neighbor closer to another real than to this fake?
    authen = real_min_dists.values[gen_min_dists.indices] < gen_min_dists.values
    return (100 * torch.sum(authen) / len(authen)).item()
```

## Vendi Score: Diversity

Diversity via eigenvalue entropy of feature similarity matrix:

```python
def compute_vendi_score(X, q=1, normalize=True):
    X = preprocessing.normalize(X, axis=1)
    S = X @ X.T
    w = scipy.linalg.eigvalsh(S / len(X))
    return np.exp(entropy_q(w, q=q))
```

## FD-infinity: Sample-Size Debiased FID

Extrapolates FID to infinite sample count via linear regression:

```python
def compute_FD_infinity(reps1, reps2, num_points=15):
    fd_batches = np.linspace(len(reps2)//10, len(reps2), num_points).astype(int)
    fds = [compute_FD(reps1, reps2[rng.choice(len(reps2), n)]) for n in fd_batches]
    reg = LinearRegression().fit(1/fd_batches.reshape(-1, 1), fds)
    return reg.predict([[0]])[0, 0]  # FD at 1/N -> 0
```
