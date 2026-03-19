from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Database
    database_url: str = "postgresql+asyncpg://seller:seller_pass@localhost:5432/motivated_seller"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Auth
    clerk_secret_key: str = ""
    clerk_publishable_key: str = ""
    secret_key: str = "dev-secret-change-me"

    # Geocoding
    geocoder: str = "nominatim"  # 'nominatim' or 'google'
    google_maps_api_key: str = ""
    nominatim_email: str = "admin@motivatedseller.app"  # Required by Nominatim ToS

    # ATTOM Data
    attom_api_key: str = ""

    # Supabase (GCP pipeline output)
    supabase_url: str = ""
    supabase_key: str = ""

    # Dallas Open Data (Socrata)
    dallas_open_data_token: str = ""

    # Proxy (Phase 3)
    proxy_host: str = ""
    proxy_port: str = ""
    proxy_user: str = ""
    proxy_pass: str = ""

    # App
    environment: str = "development"
    cors_origins: str = "http://localhost:5173"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",")]


settings = Settings()
