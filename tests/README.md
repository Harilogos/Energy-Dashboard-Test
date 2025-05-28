# Energy Generation Dashboard - Testing Documentation

This directory contains comprehensive tests for the Energy Generation Dashboard application.

## 📁 Test Structure

```
tests/
├── conftest.py                 # Shared fixtures and test configuration
├── test_runner.py             # Test runner script with various options
├── README.md                  # This documentation
├── unit/                      # Unit tests
│   ├── __init__.py
│   ├── test_config.py         # Configuration module tests
│   ├── test_data.py           # Data layer tests
│   ├── test_ui_components.py  # UI component tests
│   ├── test_visualization.py  # Visualization tests
│   └── test_services.py       # Backend services tests
└── integration/               # Integration tests
    ├── __init__.py
    └── test_app_integration.py # End-to-end integration tests
```

## 🧪 Test Categories

### Unit Tests
- **Configuration Tests** (`test_config.py`)
  - App configuration loading and validation
  - API configuration and credentials
  - Time-of-Day (ToD) slot definitions

- **Data Layer Tests** (`test_data.py`)
  - Data fetching and processing functions
  - CSV data loading and filtering
  - API data integration
  - Data transformation and aggregation
  - Caching functionality

- **UI Component Tests** (`test_ui_components.py`)
  - Client and plant filter creation
  - Date filter functionality
  - Plant selection logic
  - Error handling in UI components

- **Visualization Tests** (`test_visualization.py`)
  - Chart creation functions
  - Plot styling and formatting
  - Data visualization with various data types
  - Error handling in plotting functions

- **Services Tests** (`test_services.py`)
  - Smart data fetcher functionality
  - Cache initialization and management
  - API cache manager operations
  - Performance optimization utilities

### Integration Tests
- **App Integration Tests** (`test_app_integration.py`)
  - End-to-end application flow
  - Data pipeline integration
  - UI to data integration
  - Error handling across components
  - Complete user scenarios

## 🚀 Running Tests

### Prerequisites
Install test dependencies:
```bash
pip install -r requirements-test.txt
```

### Quick Start
Run all tests:
```bash
python tests/test_runner.py --all
```

### Specific Test Categories

**Unit Tests Only:**
```bash
python tests/test_runner.py --unit
```

**Integration Tests Only:**
```bash
python tests/test_runner.py --integration
```

**With Coverage Analysis:**
```bash
python tests/test_runner.py --all --coverage
```

**Verbose Output:**
```bash
python tests/test_runner.py --all --verbose
```

### Running Specific Tests

**Specific Test File:**
```bash
python tests/test_runner.py --test tests/unit/test_config.py
```

**Specific Test Function:**
```bash
python tests/test_runner.py --test tests/unit/test_config.py::TestAppConfig::test_default_config_structure
```

**Tests with Specific Marker:**
```bash
python tests/test_runner.py --marker unit
python tests/test_runner.py --marker integration
python tests/test_runner.py --marker slow
```

### Generate Test Report
```bash
python tests/test_runner.py --report
```

This generates:
- HTML coverage report
- XML coverage report
- JUnit XML report
- Terminal coverage summary

## 🔧 Test Configuration

### Pytest Configuration
The `pytest.ini` file contains:
- Test discovery settings
- Markers for test categorization
- Coverage configuration
- Timeout settings

### Test Markers
Available markers:
- `unit`: Unit tests
- `integration`: Integration tests
- `slow`: Slow running tests
- `api`: Tests requiring API access
- `cache`: Caching functionality tests
- `ui`: UI component tests
- `data`: Data processing tests
- `visualization`: Plotting tests
- `config`: Configuration tests
- `services`: Backend service tests

### Fixtures
Common fixtures in `conftest.py`:
- `sample_generation_data`: Mock generation data
- `sample_consumption_data`: Mock consumption data
- `sample_daily_data`: Mock daily aggregated data
- `sample_tod_data`: Mock Time-of-Day data
- `sample_client_data`: Mock client configuration
- `temp_client_json`: Temporary client JSON file
- `temp_csv_file`: Temporary CSV file
- `mock_config`: Mock configuration
- `mock_integration`: Mock API integration
- `mock_streamlit`: Mock Streamlit components
- `temp_cache_dir`: Temporary cache directory

## 📊 Test Coverage

### Coverage Goals
- **Overall Coverage**: > 80%
- **Critical Modules**: > 90%
  - Data processing functions
  - Configuration loading
  - Cache management
  - Core business logic

### Coverage Reports
After running tests with coverage:
- HTML Report: `test_reports/[timestamp]/coverage_html/index.html`
- Terminal Summary: Displayed after test run
- XML Report: `test_reports/[timestamp]/coverage.xml`

## 🛠️ Writing New Tests

### Test File Naming
- Unit tests: `test_[module_name].py`
- Integration tests: `test_[feature]_integration.py`

### Test Function Naming
- Descriptive names: `test_[function]_[scenario]`
- Examples:
  - `test_load_config_with_existing_file`
  - `test_get_generation_data_smart_success`
  - `test_create_comparison_plot_empty_data`

### Test Class Organization
```python
class TestModuleName:
    """Test cases for module_name functionality."""
    
    def test_function_success_case(self):
        """Test successful execution of function."""
        pass
    
    def test_function_error_case(self):
        """Test error handling in function."""
        pass
```

### Using Fixtures
```python
def test_with_sample_data(self, sample_generation_data):
    """Test using sample data fixture."""
    assert not sample_generation_data.empty
    assert 'generation_kwh' in sample_generation_data.columns
```

### Mocking External Dependencies
```python
@patch('module.external_dependency')
def test_with_mock(self, mock_dependency):
    """Test with mocked external dependency."""
    mock_dependency.return_value = expected_value
    result = function_under_test()
    assert result == expected_result
```

## 🐛 Debugging Tests

### Running Tests in Debug Mode
```bash
python tests/test_runner.py --test tests/unit/test_config.py --verbose
```

### Using pytest directly
```bash
pytest tests/unit/test_config.py -v -s
```

### Debug Specific Test
```bash
pytest tests/unit/test_config.py::TestAppConfig::test_default_config_structure -v -s
```

## 📈 Performance Testing

### Slow Test Identification
Tests marked with `@pytest.mark.slow` for performance monitoring:
```bash
python tests/test_runner.py --marker slow
```

### Memory Usage Testing
Some tests include memory usage validation for cache operations and data processing.

## 🔄 Continuous Integration

### Pre-commit Hooks
Run tests before commits:
```bash
pre-commit install
```

### Test Dependencies Check
```bash
python tests/test_runner.py --check-deps
```

## 📝 Best Practices

1. **Test Independence**: Each test should be independent and not rely on other tests
2. **Clear Assertions**: Use descriptive assertion messages
3. **Mock External Dependencies**: Mock API calls, file system operations, etc.
4. **Test Edge Cases**: Include tests for empty data, invalid inputs, error conditions
5. **Use Fixtures**: Leverage fixtures for common test data and setup
6. **Descriptive Names**: Use clear, descriptive test and function names
7. **Documentation**: Add docstrings to test classes and complex test functions

## 🚨 Troubleshooting

### Common Issues

**Import Errors:**
- Ensure all dependencies are installed: `pip install -r requirements-test.txt`
- Check Python path configuration

**Streamlit Import Issues:**
- Set environment variable: `STREAMLIT_SERVER_HEADLESS=true`

**Cache Directory Issues:**
- Ensure write permissions for cache directories
- Clean up test cache: `rm -rf cache/test/`

**Memory Issues with Large Tests:**
- Run tests with limited parallelization: `pytest -n 2`
- Use `pytest-xdist` for distributed testing

### Getting Help
- Check test logs in `logs/test/`
- Run with verbose output: `--verbose`
- Use pytest's built-in debugging: `--pdb`
