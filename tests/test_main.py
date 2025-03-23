import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from io import BytesIO

from app.main import app
from app.services.mockup_service import MockupService

client = TestClient(app)

# Fixtures for test data
@pytest.fixture
def mock_image_file():
    """Create a mock image file for testing"""
    return BytesIO(b"mock image content")

@pytest.fixture
def mock_mockup_service():
    """Create a mock mockup service for testing"""
    with patch("app.dependencies.get_mockup_service") as mock_get_service:
        mock_service = MagicMock(spec=MockupService)
        mock_service.generate_mockup.return_value = {
            "file_path": "output/test-mockup-id.png"
        }
        mock_get_service.return_value = mock_service
        yield mock_service

# Create test output file
@pytest.fixture
def test_output_file():
    """Create a test output file"""
    os.makedirs("output", exist_ok=True)
    with open("output/test-mockup-id.png", "wb") as f:
        f.write(b"test image data")
    yield "test-mockup-id"
    # Cleanup
    if os.path.exists("output/test-mockup-id.png"):
        os.remove("output/test-mockup-id.png")

# Tests
def test_root_endpoint():
    """Test the health check endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "message": "T-Shirt Mockup Generator API is running"}

def test_generate_mockup_success(mock_image_file, mock_mockup_service):
    """Test successful mockup generation"""
    response = client.post(
        "/mockups/generate",
        files={
            "source_image": ("image.jpg", mock_image_file, "image/jpeg"),
            "source_mask": ("mask.png", mock_image_file, "image/png"),
            "source_depth": ("depth.png", mock_image_file, "image/png"),
            "design_image": ("design.png", mock_image_file, "image/png"),
        },
        data={
            "color_code": "#FF0000",
            "location_x": 100,
            "location_y": 100,
            "scale_factor": 0.5,
            "shading_strength": 0.7,
            "color_mode": "auto",
        },
    )
    
    assert response.status_code == 200
    assert "mockup_id" in response.json()
    assert "file_path" in response.json()
    assert "download_url" in response.json()
    
    # Verify mockup service was called with correct parameters
    mock_mockup_service.generate_mockup.assert_called_once()
    call_args = mock_mockup_service.generate_mockup.call_args[1]
    assert call_args["color_code"] == "#FF0000"
    assert call_args["location"] == (100, 100)
    assert call_args["scale_factor"] == 0.5
    assert call_args["shading_strength"] == 0.7
    assert call_args["color_mode"] == "auto"

def test_generate_mockup_with_session_id(mock_image_file, mock_mockup_service):
    """Test mockup generation with a provided session ID"""
    response = client.post(
        "/mockups/generate",
        files={
            "source_image": ("image.jpg", mock_image_file, "image/jpeg"),
            "source_mask": ("mask.png", mock_image_file, "image/png"),
            "source_depth": ("depth.png", mock_image_file, "image/png"),
            "design_image": ("design.png", mock_image_file, "image/png"),
        },
        data={
            "color_code": "#FF0000",
            "location_x": 100,
            "location_y": 100,
            "scale_factor": 0.5,
            "shading_strength": 0.7,
            "color_mode": "auto",
            "session_id": "test-session-id",
        },
    )
    
    assert response.status_code == 200
    assert response.json()["mockup_id"] == "test-session-id"

def test_generate_mockup_validation_error(mock_image_file, mock_mockup_service):
    """Test mockup generation with validation error"""
    # Setup mock to raise ValueError
    mock_mockup_service.generate_mockup.side_effect = ValueError("Invalid color code")
    
    response = client.post(
        "/mockups/generate",
        files={
            "source_image": ("image.jpg", mock_image_file, "image/jpeg"),
            "source_mask": ("mask.png", mock_image_file, "image/png"),
            "source_depth": ("depth.png", mock_image_file, "image/png"),
            "design_image": ("design.png", mock_image_file, "image/png"),
        },
        data={
            "color_code": "invalid-color",
            "location_x": 100,
            "location_y": 100,
            "scale_factor": 0.5,
            "shading_strength": 0.7,
            "color_mode": "auto",
        },
    )
    
    assert response.status_code == 400
    assert "detail" in response.json()
    assert "Invalid color code" in response.json()["detail"]

def test_generate_mockup_server_error(mock_image_file, mock_mockup_service):
    """Test mockup generation with server error"""
    # Setup mock to raise Exception
    mock_mockup_service.generate_mockup.side_effect = Exception("Processing error")
    
    response = client.post(
        "/mockups/generate",
        files={
            "source_image": ("image.jpg", mock_image_file, "image/jpeg"),
            "source_mask": ("mask.png", mock_image_file, "image/png"),
            "source_depth": ("depth.png", mock_image_file, "image/png"),
            "design_image": ("design.png", mock_image_file, "image/png"),
        },
        data={
            "color_code": "#FF0000",
            "location_x": 100,
            "location_y": 100,
            "scale_factor": 0.5,
            "shading_strength": 0.7,
            "color_mode": "auto",
        },
    )
    
    assert response.status_code == 500
    assert "detail" in response.json()
    assert "Processing error" in response.json()["detail"]

def test_download_mockup_success(test_output_file):
    """Test successful mockup download"""
    response = client.get(f"/mockups/{test_output_file}/download")
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.headers["content-disposition"] == f'attachment; filename="tshirt_mockup_{test_output_file}.png"'
    assert response.content == b"test image data"

def test_download_mockup_not_found():
    """Test mockup download with non-existent ID"""
    response = client.get("/mockups/non-existent-id/download")
    
    assert response.status_code == 404
    assert "detail" in response.json()
    assert "Mockup not found" in response.json()["detail"]

def test_delete_mockup_success(test_output_file):
    """Test successful mockup deletion"""
    response = client.delete(f"/mockups/{test_output_file}")
    
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert f"Mockup {test_output_file} deleted successfully" in response.json()["message"]
    assert not os.path.exists(f"output/{test_output_file}.png")

def test_delete_mockup_not_found():
    """Test mockup deletion with non-existent ID"""
    response = client.delete("/mockups/non-existent-id")
    
    assert response.status_code == 404
    assert "detail" in response.json()
    assert "Mockup not found" in response.json()["detail"] 