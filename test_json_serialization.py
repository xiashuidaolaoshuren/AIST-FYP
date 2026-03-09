#!/usr/bin/env python
"""Test script to verify numpy array JSON serialization fix."""

import sys
import json
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

# Import the helper function from the demo script
from scripts.demo_baseline_rag import make_json_serializable


def test_json_serialization():
    """Test the make_json_serializable function with various numpy types."""
    
    # Create test data with various numpy types
    test_data = {
        'numpy_array': np.array([1, 2, 3, 4, 5]),
        'nested_dict': {
            'nested_array': np.array([[1, 2], [3, 4]]),
            'numpy_float': np.float32(3.14159),
            'numpy_int': np.int64(42),
            'numpy_bool': np.bool_(True),
            'list_of_arrays': [np.array([1, 2]), np.array([3, 4])],
        },
        'mixed_list': [
            1,
            'string',
            np.array([5, 6, 7]),
            {'key': np.float64(2.71828)}
        ]
    }
    
    print("Original test data (with numpy types):")
    print(f"  numpy_array type: {type(test_data['numpy_array'])}")
    print(f"  nested_array type: {type(test_data['nested_dict']['nested_array'])}")
    print(f"  numpy_float type: {type(test_data['nested_dict']['numpy_float'])}")
    print(f"  numpy_int type: {type(test_data['nested_dict']['numpy_int'])}")
    print(f"  numpy_bool type: {type(test_data['nested_dict']['numpy_bool'])}")
    
    # Convert to JSON-serializable
    serializable_data = make_json_serializable(test_data)
    
    print("\n✓ Converted to JSON-serializable types")
    
    # Try to dump to JSON
    try:
        json_str = json.dumps(serializable_data, indent=2)
        print("✓ Successfully serialized to JSON")
        print(f"\nJSON output (first 300 chars):\n{json_str[:300]}...")
        
        # Verify we can load it back
        loaded = json.loads(json_str)
        print("\n✓ Successfully loaded JSON back")
        
        return True
    except Exception as e:
        print(f"❌ Failed to serialize: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_json_serialization()
    sys.exit(0 if success else 1)
