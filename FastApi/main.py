from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello world"}

@app.get("/foo")
async def root():
    return {"message": "fool"}

@app.get("/items/{item_id}")
async def root(item_id: int):
    return {"item_id": item_id}

@app.post("/")
async def root():
    return {"message": "POST method"}