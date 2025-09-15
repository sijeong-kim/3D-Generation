#!/usr/bin/env python3
"""
Test script to verify that eval_radius can be parsed from settings section.
"""

import sys
import os
sys.path.append('.')

from scripts.experiments.exp_config import merge_configs

def test_eval_radius_in_settings():
    """Test that eval_radius works when defined in settings section."""
    
    # Test configuration with eval_radius in settings
    base_config = {'test': 'base'}
    fixed_params = {}
    fixed_params_dict = {}
    setting_params = {
        'eval_radius': {
            'hamburger': 3.0,
            'icecream': 4.0,
            'cactus': 4.0,
            'tulip': 4.0,
            'default': 3.5
        },
        'eval_H': 512,
        'eval_W': 512
    }
    sweep_params = {'prompt': 'hamburger'}
    sweep_params_dict = {
        'prompt': {
            'hamburger': 'a photo of a hamburger', 
            'icecream': 'a photo of an ice cream',
            'cactus': 'a small saguaro cactus planted in a clay pot',
            'tulip': 'a photo of a tulip'
        }
    }

    # Test hamburger prompt
    result = merge_configs(base_config, fixed_params, fixed_params_dict, setting_params, sweep_params, sweep_params_dict)
    
    print("=== Test 1: eval_radius in settings (hamburger) ===")
    print(f"eval_radius: {result.get('eval_radius')}")
    print(f"eval_H: {result.get('eval_H')}")
    print(f"eval_W: {result.get('eval_W')}")
    print(f"prompt: {result.get('prompt')}")
    
    assert result.get('eval_radius') == 3.0, f"Expected 3.0, got {result.get('eval_radius')}"
    assert result.get('eval_H') == 512, f"Expected 512, got {result.get('eval_H')}"
    assert result.get('eval_W') == 512, f"Expected 512, got {result.get('eval_W')}"
    assert result.get('prompt') == 'a photo of a hamburger', f"Expected 'a photo of a hamburger', got {result.get('prompt')}"
    
    # Test icecream prompt
    sweep_params_icecream = {'prompt': 'icecream'}
    result2 = merge_configs(base_config, fixed_params, fixed_params_dict, setting_params, sweep_params_icecream, sweep_params_dict)
    
    print("\n=== Test 2: eval_radius in settings (icecream) ===")
    print(f"eval_radius: {result2.get('eval_radius')}")
    print(f"prompt: {result2.get('prompt')}")
    
    assert result2.get('eval_radius') == 4.0, f"Expected 4.0, got {result2.get('eval_radius')}"
    assert result2.get('prompt') == 'a photo of an ice cream', f"Expected 'a photo of an ice cream', got {result2.get('prompt')}"
    
    # Test default fallback
    sweep_params_unknown = {'prompt': 'unknown_prompt'}
    result3 = merge_configs(base_config, fixed_params, fixed_params_dict, setting_params, sweep_params_unknown, sweep_params_dict)
    
    print("\n=== Test 3: eval_radius default fallback ===")
    print(f"eval_radius: {result3.get('eval_radius')}")
    
    assert result3.get('eval_radius') == 3.5, f"Expected 3.5 (default), got {result3.get('eval_radius')}"
    
    print("\n✅ All tests passed! eval_radius can now be parsed from settings section.")

def test_eval_radius_priority():
    """Test that fixed_parameters takes priority over settings when both are present."""
    
    base_config = {'test': 'base'}
    fixed_params = {}
    fixed_params_dict = {
        'eval_radius': {
            'hamburger': 2.0,  # Different value in fixed_params
            'icecream': 5.0,
            'default': 2.5
        }
    }
    setting_params = {
        'eval_radius': {
            'hamburger': 3.0,  # This should be ignored
            'icecream': 4.0,
            'default': 3.5
        }
    }
    sweep_params = {'prompt': 'hamburger'}
    sweep_params_dict = {
        'prompt': {
            'hamburger': 'a photo of a hamburger'
        }
    }

    result = merge_configs(base_config, fixed_params, fixed_params_dict, setting_params, sweep_params, sweep_params_dict)
    
    print("\n=== Test 4: Priority test (fixed_parameters over settings) ===")
    print(f"eval_radius: {result.get('eval_radius')}")
    
    # Should use the value from fixed_params_dict (2.0), not settings (3.0)
    assert result.get('eval_radius') == 2.0, f"Expected 2.0 (from fixed_params), got {result.get('eval_radius')}"
    
    print("✅ Priority test passed! fixed_parameters takes precedence over settings.")

if __name__ == "__main__":
    test_eval_radius_in_settings()
    test_eval_radius_priority()
    print("\n🎉 All tests completed successfully!")
