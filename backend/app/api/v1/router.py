from fastapi import APIRouter

from app.api.v1.endpoints.properties import router as properties_router, counties_router
from app.api.v1.endpoints.export import router as export_router
from app.api.v1.endpoints.scrape_jobs import router as scrape_jobs_router
from app.api.v1.endpoints.import_data import router as import_router
from app.api.v1.endpoints.status import router as status_router

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(properties_router)
api_router.include_router(counties_router)
api_router.include_router(export_router)
api_router.include_router(scrape_jobs_router)
api_router.include_router(import_router)
api_router.include_router(status_router)

from app.api.v1.endpoints.market import router as market_router
api_router.include_router(market_router)

from app.api.v1.endpoints.pipeline import router as pipeline_router
api_router.include_router(pipeline_router)

from app.api.v1.endpoints.admin import router as admin_router
api_router.include_router(admin_router)
