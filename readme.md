Architecture Overview
This FastAPI application follows a clean architecture with:
Interfaces: Abstract base classes defining the contract for implementations
Implementations: Concrete implementations of the interfaces
Services: Business logic layer that orchestrates operations
Schemas: Pydantic models for request/response validation
Dependencies: FastAPI dependency injection system
Config: Application configuration
The design makes it easy to:
Switch implementations (e.g., replace OpenCV with another library)
Test components in isolation
Migrate to Django or another framework by reusing the core business logic
Running the Application
1. Install dependencies:
```
pip install -r bulk-mockup-api/requirements.txt
```
Run the server:
```
cd bulk-mockup-api
python run.py
```
Access the API documentation at http://localhost:8000/docs
Migration to Django
To migrate to Django, you would:
Keep the interfaces, implementations, and services layers
Replace FastAPI endpoints with Django views
3. Replace FastAPI dependency injection with Django's dependency injection or a custom solution
Adapt the request/response handling to Django's patterns
The core business logic in the services and implementations would remain largely unchanged.
