"""Pacote de roteadores FastAPI (`APIRouter`)."""
from app.routers.predict_routes import health_router, predict_router

__all__ = ["health_router", "predict_router"]
