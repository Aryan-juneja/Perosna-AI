from fastapi import FastAPI, Query
from task_queue.connection import q
from worker import process_query
app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}



@app.post("/chat")
def chat(query:str = Query(..., description="The query to chat with")):
    result = q.enqueue(process_query, query)
    return {"message": "Chat request received", "job_id": result.id}


@app.get("/status/{job_id}")
def get_status(job_id: str):
    job = q.fetch_job(job_id)
    if job is None:
        return {"status": "not found"}
    return {"job_id": job.id, "status": job.get_status(), "result": job.result}