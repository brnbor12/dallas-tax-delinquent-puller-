from fastapi import APIRouter

from app.api.v1.endpoints.properties import router as properties_router
from app.api.v1.endpoints.export import router as export_router
from app.api.v1.endpoints.scrape_jobs import router as scrape_jobs_router

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(properties_router)
api_router.include_router(export_router)
api_router.include_router(scrape_jobs_router)
