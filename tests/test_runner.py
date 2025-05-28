"""
Test runner for the Energy Generation Dashboard.

This script provides utilities to run tests with different configurations
and generate test reports.
"""
import pytest
import sys
import os
from pathlib import Path
import argparse
import subprocess
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_unit_tests(verbose=False, coverage=False):
    """Run unit tests."""
    print("🧪 Running Unit Tests...")
    
    args = ["python", "-m", "pytest", "tests/unit/"]
    
    if verbose:
        args.append("-v")
    
    if coverage:
        args.extend(["--cov=backend", "--cov=frontend", "--cov=src", "--cov-report=html", "--cov-report=term"])
    
    result = subprocess.run(args, cwd=project_root)
    return result.returncode == 0


def run_integration_tests(verbose=False):
    """Run integration tests."""
    print("🔗 Running Integration Tests...")
    
    args = ["python", "-m", "pytest", "tests/integration/"]
    
    if verbose:
        args.append("-v")
    
    result = subprocess.run(args, cwd=project_root)
    return result.returncode == 0


def run_all_tests(verbose=False, coverage=False):
    """Run all tests."""
    print("🚀 Running All Tests...")
    
    args = ["python", "-m", "pytest", "tests/"]
    
    if verbose:
        args.append("-v")
    
    if coverage:
        args.extend(["--cov=backend", "--cov=frontend", "--cov=src", "--cov-report=html", "--cov-report=term"])
    
    result = subprocess.run(args, cwd=project_root)
    return result.returncode == 0


def run_specific_test(test_path, verbose=False):
    """Run a specific test file or test function."""
    print(f"🎯 Running Specific Test: {test_path}")
    
    args = ["python", "-m", "pytest", test_path]
    
    if verbose:
        args.append("-v")
    
    result = subprocess.run(args, cwd=project_root)
    return result.returncode == 0


def run_tests_by_marker(marker, verbose=False):
    """Run tests with specific marker."""
    print(f"🏷️ Running Tests with Marker: {marker}")
    
    args = ["python", "-m", "pytest", "-m", marker]
    
    if verbose:
        args.append("-v")
    
    result = subprocess.run(args, cwd=project_root)
    return result.returncode == 0


def generate_test_report():
    """Generate a comprehensive test report."""
    print("📊 Generating Test Report...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = project_root / "test_reports" / timestamp
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # Run tests with coverage and generate reports
    args = [
        "python", "-m", "pytest", "tests/",
        "--cov=backend", "--cov=frontend", "--cov=src",
        "--cov-report=html:" + str(report_dir / "coverage_html"),
        "--cov-report=xml:" + str(report_dir / "coverage.xml"),
        "--cov-report=term",
        "--junit-xml=" + str(report_dir / "junit.xml"),
        "-v"
    ]
    
    result = subprocess.run(args, cwd=project_root)
    
    if result.returncode == 0:
        print(f"✅ Test report generated in: {report_dir}")
        print(f"📁 Coverage HTML report: {report_dir / 'coverage_html' / 'index.html'}")
    else:
        print("❌ Test report generation failed")
    
    return result.returncode == 0


def check_test_dependencies():
    """Check if all test dependencies are installed."""
    print("🔍 Checking Test Dependencies...")
    
    required_packages = [
        "pytest",
        "pytest-cov",
        "pytest-mock",
        "pandas",
        "numpy",
        "matplotlib",
        "streamlit"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Install them with: pip install " + " ".join(missing_packages))
        return False
    else:
        print("✅ All test dependencies are installed")
        return True


def setup_test_environment():
    """Setup test environment."""
    print("⚙️ Setting up Test Environment...")
    
    # Set environment variables for testing
    os.environ['TESTING'] = 'true'
    os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
    
    # Create test directories if they don't exist
    test_dirs = [
        project_root / "test_reports",
        project_root / "cache" / "test",
        project_root / "logs" / "test"
    ]
    
    for test_dir in test_dirs:
        test_dir.mkdir(parents=True, exist_ok=True)
    
    print("✅ Test environment setup complete")


def cleanup_test_environment():
    """Cleanup test environment."""
    print("🧹 Cleaning up Test Environment...")
    
    # Remove test environment variables
    test_env_vars = ['TESTING', 'STREAMLIT_SERVER_HEADLESS']
    for var in test_env_vars:
        if var in os.environ:
            del os.environ[var]
    
    # Clean up test cache files
    test_cache_dir = project_root / "cache" / "test"
    if test_cache_dir.exists():
        import shutil
        shutil.rmtree(test_cache_dir, ignore_errors=True)
    
    print("✅ Test environment cleanup complete")


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Energy Dashboard Test Runner")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--test", type=str, help="Run specific test file or function")
    parser.add_argument("--marker", type=str, help="Run tests with specific marker")
    parser.add_argument("--report", action="store_true", help="Generate comprehensive test report")
    parser.add_argument("--check-deps", action="store_true", help="Check test dependencies")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--coverage", action="store_true", help="Run with coverage analysis")
    
    args = parser.parse_args()
    
    # Check dependencies first
    if args.check_deps:
        return 0 if check_test_dependencies() else 1
    
    # Setup test environment
    setup_test_environment()
    
    try:
        success = True
        
        if args.report:
            success = generate_test_report()
        elif args.unit:
            success = run_unit_tests(verbose=args.verbose, coverage=args.coverage)
        elif args.integration:
            success = run_integration_tests(verbose=args.verbose)
        elif args.test:
            success = run_specific_test(args.test, verbose=args.verbose)
        elif args.marker:
            success = run_tests_by_marker(args.marker, verbose=args.verbose)
        elif args.all:
            success = run_all_tests(verbose=args.verbose, coverage=args.coverage)
        else:
            # Default: run all tests
            success = run_all_tests(verbose=args.verbose, coverage=args.coverage)
        
        return 0 if success else 1
    
    finally:
        cleanup_test_environment()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
