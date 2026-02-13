from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.api import data, model, predict, monitor

app = FastAPI(
    title="PipeSim",
    description="Educational MLOps simulator",
    version="0.1.0"
)

# Static & templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# API routers
app.include_router(data.router, prefix="/api/data", tags=["Data"])
app.include_router(model.router, prefix="/api/model", tags=["Model"])
app.include_router(predict.router, prefix="/api/predict", tags=["Predict"])
app.include_router(monitor.router, prefix="/api/monitor", tags=["Monitor"])


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )
