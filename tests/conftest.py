import os

os.environ["DATABASE_URL"] = ""
os.environ["OLLAMA_BASE_URL"] = ""
os.environ["BDL_API_KEY"] = ""
os.environ["MODEL_ARTIFACT_PATH"] = "models/does-not-exist.joblib"
os.environ["MIN_PRIOR_GAMES"] = "5"

from courtvision.config import get_settings

get_settings.cache_clear()
