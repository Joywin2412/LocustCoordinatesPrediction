
from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
from fastapi.middleware.cors import CORSMiddleware


xgb_model = load('model/best_model.joblib')

app = FastAPI()
origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:3000",
    "http://192.168.67.39:3000",
    "https://jocular-cactus-b39620.netlify.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Item(BaseModel):
    year: int
    month: int
    day_of_year: int
    season: int


class response(BaseModel):
    coords: list[float]


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/predict", response_model=response)
def predict(arr: Item):
    year = arr.year
    month = arr.month
    day_of_year = arr.day_of_year
    season = arr.season
    print(xgb_model.predict([[year, month, day_of_year, season]]))
    ans = xgb_model.predict([[year, month, day_of_year, season]]).tolist()
    return {"coords": [ans[0][0], ans[0][1]]}
