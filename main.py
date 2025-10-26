from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.api import router

app = FastAPI(
    title = "Influencer ML Model Pipeline",
    description= "All purpose pipline which aggregates all the ml models and generates a comprehensible output.",
    version= "1.0.0"
)

# Temporary, to call api on locally run website
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://127.0.0.1:5500"] if you're serving frontend via VSCode Live Server, etc.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router)