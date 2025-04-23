from fastapi import FastAPI
from pydantic import BaseModel
# from query_engine_new4 import retrieve_and_generate_answer
from query_engine import QueryEngine


class User_input(BaseModel):
    question: str

app = FastAPI()

@app.post('/ask')
# def display(user_input: User_input):
#     return "I am doing fantastic, how can I help you today?"
# async def ask_question(request: dict):
#question = request.get("question", "")
async def ask_question(user_input: User_input):
    question = user_input.question.strip()
    if not question:
        return {"error": "No question provided"}
    
    try:
        query_engine = QueryEngine()
        response = await query_engine.retrieve_and_answer(question)
        return {"answer": response}
    except Exception as e:
        print(f" Error during processing: {e}")
        return {"answer": f"Internal error occurred: {str(e)}"}   

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)