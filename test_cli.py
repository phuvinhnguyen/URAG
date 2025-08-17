#!/usr/bin/env python3
"""
Test script to verify the CLI works correctly.
"""

import subprocess
import sys
import os

def test_help():
    """Test that help works."""
    print("Testing CLI help...")
    result = subprocess.run([sys.executable, "cli.py", "--help"], 
                          capture_output=True, text=True)
    assert result.returncode == 0, f"Help failed: {result.stderr}"
    assert "system" in result.stdout
    print("✓ Help works")

def test_list_systems():
    """Test that systems can be imported."""
    print("Testing system imports...")
    try:
        from systems import list_systems, get_system
        systems = list_systems()
        print(f"✓ Available systems: {list(systems.keys())}")
        
        # Test system creation
        system = get_system("simplellm", model_name="microsoft/DialoGPT-small")
        print("✓ SimpleLLM system created")
        
        system = get_system("simplerag", model_name="microsoft/DialoGPT-small") 
        print("✓ SimpleRAG system created")
        
    except Exception as e:
        print(f"✗ System import failed: {e}")
        return False
    
    return True

def test_evaluation():
    """Test a quick evaluation."""
    print("Testing evaluation with example dataset...")
    
    # Run a quick test
    cmd = [sys.executable, "cli.py", 
           "--system", "simplellm",
           "--dataset", "datasets/example.json",
           "--output", "test_results",
           "--model", "microsoft/DialoGPT-small"]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"✗ Evaluation failed: {result.stderr}")
        print(f"Stdout: {result.stdout}")
        return False
    
    print("✓ Evaluation completed successfully")
    
    # Check that output files were created
    if os.path.exists("test_results"):
        files = os.listdir("test_results")
        json_files = [f for f in files if f.endswith('.json')]
        if len(json_files) >= 3:  # Should have calibration, test, and metrics files
            print(f"✓ Output files created: {json_files}")
        else:
            print(f"✗ Expected at least 3 output files, got {len(json_files)}")
            return False
    else:
        print("✗ Output directory not created")
        return False
    
    return True

def main():
    """Run all tests."""
    print("Running URAG CLI tests...")
    print("="*50)
    
    tests = [
        ("Help", test_help),
        ("System Imports", test_list_systems),
        ("Evaluation", test_evaluation)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            failed += 1
    
    print("\n" + "="*50)
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! URAG is ready to use.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
