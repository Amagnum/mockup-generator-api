from fastapi import Depends

from app.interfaces.mockup_generator import MockupGenerator
from app.implementations.cv_mockup_generator import CVMockupGenerator
from app.services.mockup_service import MockupService

def get_mockup_generator() -> MockupGenerator:
    """Dependency for getting the mockup generator implementation"""
    return CVMockupGenerator()

def get_mockup_service(
    mockup_generator: MockupGenerator = Depends(get_mockup_generator)
) -> MockupService:
    """Dependency for getting the mockup service"""
    return MockupService(mockup_generator) 