"""
Test validation script to ensure the testing framework is working correctly.

This script performs basic validation of the test setup and can be run
to verify that the testing environment is properly configured.
"""
import sys
import os
from pathlib import Path
import importlib.util

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_imports():
    """Test that all required modules can be imported."""
    print("🔍 Testing module imports...")
    
    required_modules = [
        'pytest',
        'pandas',
        'numpy',
        'matplotlib',
        'streamlit',
        'unittest.mock'
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"  ✅ {module}")
        except ImportError as e:
            print(f"  ❌ {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        return False
    else:
        print("\n✅ All required modules imported successfully")
        return True


def test_project_structure():
    """Test that the project structure is correct."""
    print("\n🏗️ Testing project structure...")
    
    required_paths = [
        'backend',
        'backend/config',
        'backend/data',
        'backend/services',
        'backend/utils',
        'backend/api',
        'frontend',
        'frontend/components',
        'src',
        'tests',
        'tests/unit',
        'tests/integration',
        'tests/conftest.py',
        'app.py',
        'pytest.ini'
    ]
    
    missing_paths = []
    
    for path in required_paths:
        full_path = project_root / path
        if full_path.exists():
            print(f"  ✅ {path}")
        else:
            print(f"  ❌ {path}")
            missing_paths.append(path)
    
    if missing_paths:
        print(f"\n❌ Missing paths: {', '.join(missing_paths)}")
        return False
    else:
        print("\n✅ Project structure is correct")
        return True


def test_configuration_files():
    """Test that configuration files are valid."""
    print("\n⚙️ Testing configuration files...")
    
    # Test pytest.ini
    pytest_ini = project_root / 'pytest.ini'
    if pytest_ini.exists():
        print("  ✅ pytest.ini exists")
        try:
            with open(pytest_ini, 'r') as f:
                content = f.read()
                if '[tool:pytest]' in content:
                    print("  ✅ pytest.ini has valid format")
                else:
                    print("  ❌ pytest.ini missing [tool:pytest] section")
                    return False
        except Exception as e:
            print(f"  ❌ Error reading pytest.ini: {e}")
            return False
    else:
        print("  ❌ pytest.ini not found")
        return False
    
    # Test conftest.py
    conftest = project_root / 'tests' / 'conftest.py'
    if conftest.exists():
        print("  ✅ conftest.py exists")
        try:
            spec = importlib.util.spec_from_file_location("conftest", conftest)
            conftest_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(conftest_module)
            print("  ✅ conftest.py is valid Python")
        except Exception as e:
            print(f"  ❌ Error importing conftest.py: {e}")
            return False
    else:
        print("  ❌ conftest.py not found")
        return False
    
    print("\n✅ Configuration files are valid")
    return True


def test_sample_fixtures():
    """Test that sample fixtures work correctly."""
    print("\n🧪 Testing sample fixtures...")
    
    try:
        # Import conftest to access fixtures
        sys.path.insert(0, str(project_root / 'tests'))
        import conftest
        
        # Test sample data generation
        generator = conftest.TestDataGenerator()
        
        # Test time series generation
        ts_data = generator.generate_time_series('2024-01-01', '2024-01-02')
        if not ts_data.empty and 'time' in ts_data.columns:
            print("  ✅ Time series generation works")
        else:
            print("  ❌ Time series generation failed")
            return False
        
        # Test plant data generation
        plant_data = generator.generate_plant_data('TEST.001', '2024-01-01', '2024-01-02')
        if not plant_data.empty and 'generation_kwh' in plant_data.columns:
            print("  ✅ Plant data generation works")
        else:
            print("  ❌ Plant data generation failed")
            return False
        
        print("\n✅ Sample fixtures work correctly")
        return True
        
    except Exception as e:
        print(f"\n❌ Error testing fixtures: {e}")
        return False


def test_basic_test_execution():
    """Test that a basic test can be executed."""
    print("\n🚀 Testing basic test execution...")
    
    try:
        import subprocess
        
        # Run a simple test to verify pytest works
        result = subprocess.run([
            sys.executable, '-m', 'pytest', 
            str(project_root / 'tests' / 'conftest.py'),
            '--collect-only', '-q'
        ], capture_output=True, text=True, cwd=project_root)
        
        if result.returncode == 0:
            print("  ✅ Pytest can collect tests")
        else:
            print(f"  ❌ Pytest collection failed: {result.stderr}")
            return False
        
        print("\n✅ Basic test execution works")
        return True
        
    except Exception as e:
        print(f"\n❌ Error testing basic execution: {e}")
        return False


def test_environment_setup():
    """Test that the test environment can be set up correctly."""
    print("\n🌍 Testing environment setup...")
    
    try:
        # Test environment variable setting
        os.environ['TESTING'] = 'true'
        if os.getenv('TESTING') == 'true':
            print("  ✅ Environment variables work")
        else:
            print("  ❌ Environment variables not working")
            return False
        
        # Test directory creation
        test_dir = project_root / 'test_temp'
        test_dir.mkdir(exist_ok=True)
        if test_dir.exists():
            print("  ✅ Directory creation works")
            test_dir.rmdir()  # Clean up
        else:
            print("  ❌ Directory creation failed")
            return False
        
        # Clean up environment
        del os.environ['TESTING']
        
        print("\n✅ Environment setup works correctly")
        return True
        
    except Exception as e:
        print(f"\n❌ Error testing environment setup: {e}")
        return False


def main():
    """Run all validation tests."""
    print("🧪 Energy Dashboard Test Validation")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_project_structure,
        test_configuration_files,
        test_sample_fixtures,
        test_basic_test_execution,
        test_environment_setup
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n❌ Test {test.__name__} failed with exception: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Validation Results:")
    print(f"  ✅ Passed: {passed}")
    print(f"  ❌ Failed: {failed}")
    print(f"  📈 Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 All validation tests passed! The testing framework is ready to use.")
        print("\nNext steps:")
        print("  1. Install test dependencies: pip install -r requirements-test.txt")
        print("  2. Run unit tests: python tests/test_runner.py --unit")
        print("  3. Run all tests: python tests/test_runner.py --all")
        return 0
    else:
        print(f"\n⚠️ {failed} validation test(s) failed. Please fix the issues before running tests.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
