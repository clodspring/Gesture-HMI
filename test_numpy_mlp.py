"""Dependency-light numeric invariant tests; no PyTorch needed for end users."""
from pathlib import Path
import importlib.util
import numpy as np

path = Path(__file__).with_name('gesture_pointer_step19_v3b_portable.py')
spec = importlib.util.spec_from_file_location('gesture_hmi', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
weights, mean, std, clip, classes = m.load_numpy_mlp()
x = np.random.default_rng(145).normal(size=(1500, 42)).astype(np.float32)
ps = m.numpy_mlp_probs(x, weights)
assert ps.shape == (1500,4)
assert ps.dtype == np.float32
assert np.all(np.isfinite(ps))
assert np.max(np.abs(ps.sum(axis=1) - 1.0)) < 1e-6
assert set(classes) == {'peace','one','stop','no_gesture'}
assert mean.shape == (42,) and std.shape == (42,)
print('1500 NUMPY MLP NUMERIC INVARIANTS PASSED')
