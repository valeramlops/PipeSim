import pytest
from httpx import AsyncClient, ASGITransport
from typing import AsyncGenerator

# Import app and Database dependency
from app.main import app
from app.database import get_db

# 1. Database mocking
# Create fake dependency to prevent tests from accessing the real database
async def override_get_db():
    try:
        # In future, you can use SQLite in memory
        yield "Mocked Database Session"
    finally:
        pass

# Tell FastAPI to replace the real get_db function with stub
app.dependency_overrides[get_db] = override_get_db

# 2. Setting async client for tests
@pytest.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """
    This fixture creates a fake browser (client) that can send requests to FastAPI application directly in memory
    """
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver"
    ) as client:
        yield client