from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.api import data, model, predict, monitor

app = FastAPI(
    title="PipeSim",
    description="Educational MLOps simulator",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Simple handler errors and validation
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # Get first error from list
    error = exc.errors()[0]

    # Get the field name (it is always at the end of the 'loc' list)
    field_name = error.get("loc")[-1]
    error_message = error.get("msg")

    # Return a clear response with status 422
    return JSONResponse(
        status_code=422,
        content={
            "status": "error",
            "message": f"Error in field '{field_name}': {error_message}"
        }
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
