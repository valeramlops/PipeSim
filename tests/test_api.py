from fastapi.testclient import TestClient
from app.main import app
from app.core.database import get_db

# 1. Database Mock

class MockSession:
    async def execute(self, query):
        # Fake answer for searching in cache
        class MockResult:
            def scalars(self):
                class MockScalars:
                    def first(self): return None
                    def all(self): return []
                return MockScalars()
        return MockResult()
    
    def add(self, instance): pass
    def add_all(self, instances): pass

    async def commit(self): pass
    async def flush(self): pass

    async def refresh(self, instance):
        # Pretend that the database has returned an ID = 999
        instance.id = 999

# Function, which get fake database
async def override_get_db():
    yield MockSession()

# Using fake database instead of real get_db
app.dependency_overrides[get_db] = override_get_db

# Create test client
client = TestClient(app)

# 2. Tests (Unit tests)
def test_predict_status():
    """
    Test: Check that status endpoint return 200 OK
    """
    response = client.post("/api/predict/status")

    assert response.status_code == 200
    assert "status" in response.json()

def test_validation_negative_age():
    """
    Test: Check validation
    """
    bad_data = {
        "Pclass": 1,
        "Sex": "male",
        "Age": -10, # Error
        "SibSp": 0,
        "Parch": 0,
        "Fare": 10.0,
        "Embarked": "S"
    }
    response = client.post("/api/predict/", json=bad_data)

    assert response.status_code == 422

    # Check that custom error return beautifully
    response_data = response.json()
    assert response_data["status"] == "error"
    assert "Age" in response_data["message"]

def test_validation_bad_pclass():
    """
    Test: Check protection from wrong passenger class
    """
    bad_data = {
        "Pclass": 5, # Error
        "Sex": "male",
        "Age": 25,
        "SibSp": 0,
        "Parch": 0,
        "Fare": 10.0,
        "Embarked": "S"
    }
    response = client.post("/api/predict/", json=bad_data)

    assert response.status_code == 422
    assert "Pclass" in response.json()["message"]

def test_predict_success():
    """
    Test: Check success predict (and fake database work)
    """
    good_data = {
        "Pclass": 1,
        "Sex": "female",
        "Age": 25,
        "SibSp": 1,
        "Parch": 0,
        "Fare": 50.0,
        "Embarked": "C"
    }
    response = client.post("/api/predict/", json=good_data)

    # Request must be success
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "success"
    assert data["prediction_id"] == 999 # Fake ID from MockSession
    assert "survived" in data
    assert "probability" in data