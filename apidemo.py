from Forex_data import EURUSD_Historical_Forex_2022_2024_processed
from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
async def read_root():
    return {"well if its working"}
@app.get("/get_demo_data")
async def read_items():
    return EURUSD_Historical_Forex_2022_2024_processed


if __name__=="_main_":
    uvicorn.run(app)
