#!/usr/bin/env python3
"""
Test script to verify the updated parsing functions work correctly.
"""

import re

def parse_config_name(config_name):
    """Parse config name to extract parameters."""
    # New pattern: REPULSION__KERNEL__λVALUE__PROMPT__S{SEED}
    # e.g., RLSD__COS__λ100__CACT__S42
    pattern = r'(\w+)__(\w+)__λ([\dK.]+)__(\w+)__S(\d+)'
    match = re.match(pattern, config_name)
    if match:
        # Convert lambda value back to numeric format
        lambda_str = match.group(3)
        if lambda_str.endswith('K'):
            lambda_value = str(int(float(lambda_str[:-1]) * 1000))
        else:
            lambda_value = lambda_str
            
        return {
            'repulsion_type': match.group(1).lower(),  # RLSD -> rlsd
            'kernel_type': match.group(2).lower(),     # COS -> cosine, RBF -> rbf
            'lambda_repulsion': lambda_value,
            'prompt': match.group(4).lower(),          # CACT -> cactus
            'seed': match.group(5)
        }
    return None

def test_parsing():
    """Test the parsing function with various config names."""
    test_configs = [
        'RLSD__COS__λ100__CACT__S42',
        'SVGD__RBF__λ1K__HAMB__S42', 
        'RLSD__COS__λ2K__ICE__S42',
        'SVGD__RBF__λ500__TUL__S42',
        'RLSD__COS__λ1.5K__CACT__S42',  # Test decimal K values
    ]
    
    print("Testing config name parsing:")
    print("=" * 50)
    
    for config in test_configs:
        result = parse_config_name(config)
        if result:
            print(f"✓ {config}")
            print(f"  -> repulsion_type: {result['repulsion_type']}")
            print(f"  -> kernel_type: {result['kernel_type']}")
            print(f"  -> lambda_repulsion: {result['lambda_repulsion']}")
            print(f"  -> prompt: {result['prompt']}")
            print(f"  -> seed: {result['seed']}")
        else:
            print(f"✗ {config} -> Failed to parse")
        print()

if __name__ == "__main__":
    test_parsing()
