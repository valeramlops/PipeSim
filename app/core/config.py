from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Project name
    PROJECT_NAME: str = "PipeSim API"

    # Connecting to DB
    DATABASE_URL: str

    # ML-model path
    MODEL_PATH: str = "models/titanic_model.pkl"

    # Reading settings from file .env
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

# Creating settins object
settings = Settings()