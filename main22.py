from fastapi import FastAPI

app = FastAPI()

@app.get("/add")
async def add_numbers(num1: float, num2: float):
    """
    This endpoint takes two numbers as query parameters and returns their sum.
    """
    result = num1 + num2
    return {"result": result}