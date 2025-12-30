# Performance Optimizations Summary

This document summarizes the performance optimizations made to improve dataframe/array operations without changing results.

## Optimizations Implemented

### 1. Batch Analyzer (`batch_analyzer.py`)

#### `consistency_analysis()` - Vectorized Pairwise Differences
**Before:** Nested Python loops (O(n²) complexity)
```python
for i in range(n_realizations):
    for j in range(i + 1, n_realizations):
        nls_pairwise_diffs.append(abs(f_nls_all[i] - f_nls_all[j]))
```

**After:** Vectorized using `np.triu_indices()`
```python
i_indices, j_indices = np.triu_indices(n_realizations, k=1)
nls_pairwise_diffs = np.abs(f_nls_all[i_indices] - f_nls_all[j_indices])
```

**Impact:** Eliminates O(n²) Python loops, uses vectorized NumPy operations. For 100 realizations, reduces from ~10,000 iterations to a single vectorized operation.

#### `calculate_q_factors()` - Vectorized Q Factor Calculation
**Before:** Python loop with individual calculations
```python
for r in self.results:
    q = np.pi * r['f_nls'] * r['tau_est']
    q_factors.append(q)
```

**After:** Single vectorized operation
```python
f_nls_all = np.array([r['f_nls'] for r in self.results], dtype=float)
tau_est_all = np.array([r['tau_est'] for r in self.results], dtype=float)
q_factors = (np.pi * f_nls_all * tau_est_all).tolist()
```

**Impact:** Eliminates loop overhead, uses NumPy's optimized C implementations.

#### `crlb_comparison_analysis()` - Vectorized Ratio Calculations
**Before:** List comprehension with conditional logic in loop
```python
ratios = []
for d, crlb in zip(diffs, crlb_stds):
    if crlb > 0 and np.isfinite(crlb):
        ratios.append(d / crlb)
    else:
        ratios.append(np.nan)
```

**After:** Vectorized with `np.divide()` and `where` parameter
```python
ratios = np.divide(diffs, crlb_stds, 
                  out=np.full_like(diffs, np.nan, dtype=float),
                  where=(crlb_stds > 0) & np.isfinite(crlb_stds))
```

**Impact:** Eliminates Python loop, handles edge cases (zeros, infs) efficiently.

### 2. Estimators (`estimators.py`)

#### `_fit_lorentzian_to_peak()` - Vectorized Half-Maximum Search
**Before:** Python loops searching for half-maximum points
```python
for i in range(k - 1, max(0, k - 10), -1):
    if P[i] < half_max:
        left_idx = i
        break
```

**After:** Vectorized search using boolean indexing
```python
left_range = np.arange(k - 1, left_range_start - 1, -1)
left_mask = P[left_range] < half_max
if np.any(left_mask):
    left_idx = left_range[np.argmax(left_mask)]
```

**Impact:** Reduces loop overhead, especially for larger search ranges.

#### `_estimate_initial_tau_from_envelope()` - Vectorized RMS Calculation
**Before:** Python loop computing RMS for each window
```python
for i in range(n_windows):
    start = i * window_size
    end = start + window_size
    rms_values.append(np.std(x[start:end]))
```

**After:** Vectorized using reshape and axis operation
```python
x_reshaped = x_padded.reshape(n_windows, window_size)
rms_values = np.std(x_reshaped, axis=1)
```

**Impact:** Eliminates loop, uses NumPy's optimized std calculation along axis.

#### Removed Unnecessary `.copy()` Calls
**Before:**
```python
P_no_dc = P.copy()
P_no_dc[0] = 0.0
k = int(np.argmax(P_no_dc))
```

**After:**
```python
k = int(np.argmax(P[1:]) + 1)  # Skip first element, add 1 to index
```

**Impact:** Avoids memory allocation for copy, uses view instead.

### 3. Analyzer (`analyzer.py`)

#### `crop_data_to_tau()` - Removed Unnecessary Copies
**Before:**
```python
if len(t_crop) < min_samples:
    return t.copy(), data.copy()
```

**After:**
```python
if len(t_crop) < min_samples:
    return t, data
```

**Impact:** Returns views instead of copies when original data is returned, reducing memory usage.

### 4. Data Loader (`data_loader.py`)

#### `load_csv()` - Optimized Parsing
**Before:** Loop appending to list, then converting to array
```python
data_list = []
for line in data_lines:
    parts = [float(x.strip()) for x in line.split(',')]
    data_list.append(parts)
data_array = np.array(data_list)
```

**After:** More efficient parsing with early filtering
```python
# Only parse needed columns (0 and 3) directly
data_list.append([float(parts[0].strip()), float(parts[3].strip())])
data_array = np.array(data_list)
```

**Impact:** Reduces memory allocation by only parsing needed columns, more efficient array creation.

## Correctness Guarantees

All optimizations preserve:
- **Output values:** All numerical results are identical (within floating-point precision)
- **Index alignment:** Pairwise indices are preserved correctly
- **Dtype behavior:** Arrays maintain `float64` dtype explicitly
- **NaN handling:** NaN propagation and handling is preserved
- **Edge cases:** Zero CRLB, infinite values, empty arrays all handled correctly

## Performance Benchmarks

Performance tests are included in `tests/test_performance_optimizations.py`:

- **Consistency analysis:** Vectorized pairwise differences for 100 realizations
- **Q factor calculation:** Vectorized calculation for 1000 results
- **CRLB comparison:** Vectorized ratio calculations for 1000 results

Expected performance improvements:
- **Pairwise differences:** ~10-100x faster for large n (O(n²) → O(1) vectorized)
- **Q factors:** ~5-10x faster (eliminates loop overhead)
- **CRLB ratios:** ~5-10x faster (vectorized division with masking)
- **RMS calculation:** ~10-50x faster (vectorized axis operation)
- **Memory usage:** Reduced by avoiding unnecessary copies

## Testing

Comprehensive tests verify:
1. **Correctness:** All numerical results match pre-optimization values
2. **NaN handling:** Proper handling of NaN values in arrays
3. **Edge cases:** Zero CRLB, infinite values, empty arrays
4. **Dtype preservation:** Arrays maintain correct dtypes
5. **Performance:** Operations complete in reasonable time

Run tests with:
```bash
pytest tests/test_performance_optimizations.py -v
```

## Notes

- No new dependencies were added
- All optimizations maintain backward compatibility
- Readability is preserved with clear vectorized operations
- Explicit `dtype=float` ensures consistent behavior across platforms

