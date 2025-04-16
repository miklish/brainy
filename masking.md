```python
# Example with mini_batch_size=4, num_experts=6, current_top_k=3
# Original gating network output (after softmax):
gating_outputs = torch.tensor([
    [0.30, 0.20, 0.10, 0.25, 0.10, 0.05],  # Sample 1
    [0.05, 0.45, 0.30, 0.05, 0.05, 0.10],  # Sample 2 
    [0.10, 0.10, 0.05, 0.50, 0.15, 0.10],  # Sample 3
    [0.25, 0.05, 0.20, 0.10, 0.10, 0.30]   # Sample 4
])

# Get top-3 experts for each sample
_, top_k_indices = torch.topk(gating_outputs, k=3, dim=1)
# top_k_indices = [[0, 3, 1], [1, 2, 5], [3, 4, 0], [0, 5, 2]]

# Create and apply mask
mask = torch.zeros_like(gating_outputs)
mask.scatter_(dim=1, index=top_k_indices, value=1.0)
gating_outputs = gating_outputs * mask

# After masking:
# [
#   [0.30, 0.20, 0.00, 0.25, 0.00, 0.00],  # Sample 1 - sum = 0.75
#   [0.00, 0.45, 0.30, 0.00, 0.00, 0.10],  # Sample 2 - sum = 0.85
#   [0.10, 0.00, 0.00, 0.50, 0.15, 0.00],  # Sample 3 - sum = 0.75
#   [0.25, 0.00, 0.20, 0.00, 0.00, 0.30]   # Sample 4 - sum = 0.75
# ]

# Calculate sums
gating_outputs_sum = gating_outputs.sum(dim=1, keepdim=True)
# gating_outputs_sum = [[0.75], [0.85], [0.75], [0.75]]
```