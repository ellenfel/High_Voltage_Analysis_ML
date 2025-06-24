## DataFrame Loop Optimization - Quick Reference

## Problem: Memory Crashes on Large DataFrames
**Root Cause:** For loops create temporary Series objects repeatedly (31M rows × 11 iterations = massive memory usage)

## The Fix: Replace Loops with Vectorized Operations

### Before (Memory Killer):
```python
for original_str, mapped_val in conversion_map.items():
    mask = df['clean_value'].isna() & (df['value'].astype(str).str.strip().str.lower() == original_str)
    df.loc[mask, 'clean_value'] = mapped_val
```

### After (Memory Efficient):
```python
# 1. Ensure deep copy (prevents data corruption)
df['clean_value'] = df['value'].copy(deep=True)

# 2. Single vectorized operation instead of loop
nan_mask = df['clean_value'].isna()  # Create mask once
temp_mapped_values = df.loc[nan_mask, 'value'].astype(str).str.strip().str.lower().map(conversion_map)
df.loc[nan_mask, 'clean_value'] = temp_mapped_values

# 3. Clean up memory
del temp_mapped_values
gc.collect()
```

## Key Changes:
1. **Deep Copy**: `copy(deep=True)` prevents view/copy issues
2. **Single Mask**: Create boolean mask once, not in every loop iteration  
3. **Subset Processing**: Only process rows that need mapping, not entire DataFrame
4. **Vectorized Map**: Use pandas `.map()` instead of string comparisons in loops
5. **Memory Cleanup**: Explicitly delete temp variables and garbage collect

## Why This Works:
- **Before**: 31M × 11 temporary string operations = memory explosion
- **After**: Process only NaN rows once = minimal memory footprint

Apply this pattern to any large DataFrame with mapping operations.