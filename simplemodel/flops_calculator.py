def compute_flops(input_dim, output_dim):
    return input_dim * output_dim * 2

def calculate_flops(hidden_states, utilized_layers):
    flops_per_layer = []
    for layer in utilized_layers:
        input_dim, output_dim = hidden_states[layer].shape[-2], hidden_states[layer].shape[-1]
        flops_per_layer.append(compute_flops(input_dim, output_dim))
    return sum(flops_per_layer)
