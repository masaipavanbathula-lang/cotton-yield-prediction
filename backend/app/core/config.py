from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    app_name: str = "Cotton Yield Prediction API"
    frontend_origins: str = "http://localhost:5173"
    data_path: str = "backend/data/cotton_growth_cycle_data.csv"
    model_dir: str = "backend/models"
    model_version: str = "1.0.0"
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    @property
    def origins(self): return [x.strip() for x in self.frontend_origins.split(',') if x.strip()]
    @property
    def data_file(self):
        p=Path(self.data_path)
        if p.exists(): return p
        return Path(__file__).resolve().parents[3] / self.data_path
    @property
    def models_path(self):
        p=Path(self.model_dir)
        if p.exists(): return p
        return Path(__file__).resolve().parents[3] / self.model_dir
@lru_cache
def get_settings(): return Settings()
