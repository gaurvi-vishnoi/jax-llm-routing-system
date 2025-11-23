import os
from openai import OpenAI
from config import OPENAI_API_KEY

"""
This script uploads finetune_data.jsonl and starts an OpenAI fine-tuning job.
The resulting model ID is saved inside llm/finetune_job.txt
"""

DATASET_PATH = "llm/finetune_data.jsonl"
OUT_JOB_FILE = "llm/finetune_job.txt"

client = OpenAI(api_key=OPENAI_API_KEY)


def upload_dataset():
    print("Uploading dataset:", DATASET_PATH)

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    file = client.files.create(
        file=open(DATASET_PATH, "rb"),
        purpose="fine-tune"
    )
    print("File uploaded:", file.id)
    return file.id


def start_finetune(file_id):
    print("Starting fine-tuning job")

    job = client.fine_tuning.jobs.create(
        training_file=file_id,
        model="gpt-4o-mini",   
        suffix="support-router"
    )

    print("Fine-tune job created!")
    print("Job ID:", job.id)

    # Save job id
    with open(OUT_JOB_FILE, "w") as f:
        f.write(job.id)

    print("Saved job ID to:", OUT_JOB_FILE)
    return job.id


def main():
    file_id = upload_dataset()
    job_id = start_finetune(file_id)
    print("\nUse create_finetune.py to check status:")
    print(f"   python -m llm.create_finetune --status {job_id}")


if __name__ == "__main__":
    main()
