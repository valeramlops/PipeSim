from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import uuid
import time

from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.api import data, model, predict, monitor, auth, vision
from app.core.logger import log
from app.core.limiter import limiter
from app.database import engine, Base
import app.models # noqa: F401

from contextlib import asynccontextmanager

# Lifespan cycle func
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Logger
    log.info("Connecting to DB and table checking")
    async with engine.begin() as conn:
        # Automatically create tables based on models
        await conn.run_sync(Base.metadata.create_all)
    log.info("Database is ready")

    yield # This is where the server is running and accepting requests
    
    log.info("Closing connecting with Database...")
    await engine.dispose()

app = FastAPI(
    title="PipeSim",
    description="Educational MLOps simulator",
    version="0.1.0",
    lifespan=lifespan
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# MIDDLEWARE
@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Generating unique ID for every request
    request_id = str(uuid.uuid4())

    # Timer
    start_time = time.time()

    # Bind request_id
    request_log = log.bind(request_id=request_id, method=request.method, path=request.url.path)
    
    request_log.info("Request started")

    try:
        # Throw request to endpoints
        response = await call_next(request)

        # Stop timer
        process_time = time.time() - start_time

        # Logging success end
        request_log.info(
            "Request compleated",
            status_code=response.status_code,
            duration_seconds=round(process_time, 4)
        )

        # Add ID to request's title
        response.headers["X-Request-ID"] = request_id
        return response
    
    except Exception as e:
        process_time = time.time() - start_time
        request_log.error(
            "Request failed (Internal Server Error)",
            error=str(e),
            duration_seconds=round(process_time, 4),
            exc_info=True # Write all red text of error (Traceback) into log
        )
        raise # Forward rhe error so that FastAPI returns a 500 status anyway

# Simple handler errors and validation
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):

    # Get first error from list
    error = exc.errors()[0]

    # Get the field name (it is always at the end of the 'loc' list)
    field_name = error.get("loc")[-1]
    error_message = error.get("msg")

    # Logging user error as Warning
    log.warning(
        "Validation Error",
        field=field_name,
        error=error_message,
        path=request.url.path
    )

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
app.include_router(vision.router, prefix="/api/vision", tags=["Computer Vision"])
app.include_router(auth.router, prefix="/api/auth", tags=["Auth"])
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
