import pytest
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Setup any global fixtures or configuration for tests here
@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment variables and configurations"""
    # Set environment variables for testing
    os.environ["DEBUG"] = "True"
    os.environ["TESTING"] = "True"
    os.environ["OUTPUT_DIR"] = "output"
    os.environ["UPLOAD_DIR"] = "uploads"
    
    # Create test directories if they don't exist
    os.makedirs("output", exist_ok=True)
    os.makedirs("uploads", exist_ok=True)
    
    yield
    
    # Cleanup could go here if needed 