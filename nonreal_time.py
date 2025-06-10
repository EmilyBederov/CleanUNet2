# Quick analysis script
def analyze_model_for_realtime(model):
    # Check for non-causal operations
    non_causal_ops = []
    
    for name, module in model.named_modules():
        # Bidirectional RNNs/LSTMs
        if isinstance(module, nn.LSTM) and module.bidirectional:
            non_causal_ops.append(f"Bidirectional LSTM at {name}")
        
        # Future-looking attention
        if "attention" in name.lower():
            # Check if attention looks at future frames
            non_causal_ops.append(f"Possible non-causal attention at {name}")
        
        # Large receptive fields
        if isinstance(module, nn.Conv1d):
            if module.kernel_size[0] > 256:  # Example threshold
                non_causal_ops.append(f"Large kernel at {name}: {module.kernel_size}")
    
    return non_causal_ops

# For CleanUNet2 specifically
print("CleanUNet2 issues:")
print(analyze_model_for_realtime(cleanunet2))