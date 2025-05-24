import os
import time
from openai import OpenAI
from dotenv import load_dotenv

# Load env vars from .env (ensure you have OPENAI_API_KEY defined there)
load_dotenv()

# Instantiate the client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment. Did you create a .env and call load_dotenv()?")

client = OpenAI(api_key=api_key)

# 1) Upload your JSONL training file
print("Uploading training data…")
upload_resp = client.files.create(
    file=open("ft_data.jsonl", "rb"),
    purpose="fine-tune"
)
training_file_id = upload_resp.id
print(f"➡ File uploaded. ID = {training_file_id!r}")

# 2) Create the fine-tune job with hyperparameters
print("Creating fine-tune job…")
ft_resp = client.fine_tuning.jobs.create(
    training_file=training_file_id,
    model="gpt-4o-2024-08-06",
    hyperparameters={
        "n_epochs": 2,
        "learning_rate_multiplier": 0.1
    }
)
ft_job_id = ft_resp.id
print(f"➡ Fine-tune job created. Job ID = {ft_job_id!r}")

# 3) Poll until completion
print("Polling for job completion…")
while True:
    status_resp = client.fine_tuning.jobs.retrieve(ft_job_id)
    status = status_resp.status
    print(f"  • status: {status}")
    if status in ("succeeded", "failed"):
        break
    time.sleep(30)

# 4) Report outcome or dump debug info
if status == "succeeded":
    custom_model = status_resp.fine_tuned_model
    print(f"🎉 Fine-tune succeeded! Your new model is: {custom_model!r}")
else:
    print("❌ Fine-tune failed. Dumping debug info…")
    # 1) Raw job payload for metadata inspection
    print("--- JOB RESPONSE ---")
    print(status_resp)
    print("--------------------\n")

    # 2) List and download any result / error files
    for f in status_resp.result_files or []:
        print(f" • Found result file: {f.filename} (id={f.id})")
        if "fail" in f.filename.lower() or "error" in f.filename.lower():
            print(f"\nDownloading error log ({f.id})…")
            err_bytes = client.files.download(f.id)
            err_text = err_bytes.decode("utf-8")
            print("----- BEGIN ERROR LOG -----")
            print(err_text)
            print("------ END ERROR LOG ------\n")

    print("Check these messages to identify and fix the issue in your JSONL.")
#🎉 Fine-tune succeeded! Your new model is: 'ft:gpt-4o-2024-08-06:viscom::BZll9oDU'
