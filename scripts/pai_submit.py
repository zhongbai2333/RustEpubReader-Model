# 文件职责：阿里云 PAI-DLC 训练任务的提交、监控与模型下载辅助脚本。
"""
PAI-DLC training job submission and monitoring helper.

Subcommands:
    upload-data    Upload datasets to OSS
    submit         Submit a PAI-DLC training job
    wait           Wait for a training job to complete
    download-model Download trained model from OSS

Environment variables required:
    ALIBABA_ACCESS_KEY_ID
    ALIBABA_ACCESS_KEY_SECRET
    ALIBABA_REGION
    ALIBABA_OSS_BUCKET
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path


def get_oss_bucket():
    """Create OSS bucket client."""
    import oss2

    key_id = os.environ["ALIBABA_ACCESS_KEY_ID"]
    key_secret = os.environ["ALIBABA_ACCESS_KEY_SECRET"]
    region = os.environ["ALIBABA_REGION"]
    bucket_name = os.environ["ALIBABA_OSS_BUCKET"]

    auth = oss2.Auth(key_id, key_secret)
    endpoint = f"https://oss-{region}.aliyuncs.com"
    return oss2.Bucket(auth, endpoint, bucket_name)


def upload_data(args):
    """Upload training datasets to OSS."""
    bucket = get_oss_bucket()
    data_dir = Path(args.data_dir)

    oss_prefix = "csc-training/datasets/"
    count = 0

    for subdir in ["base", "submissions"]:
        folder = data_dir / subdir
        if not folder.exists():
            continue
        for fpath in folder.glob("*.jsonl"):
            oss_key = f"{oss_prefix}{subdir}/{fpath.name}"
            print(f"Uploading: {fpath} → oss://{oss_key}")
            bucket.put_object_from_file(oss_key, str(fpath))
            count += 1

    # Also upload training config and scripts
    for f in ["config/training_config.yaml", "scripts/train.py", "requirements.txt"]:
        if Path(f).exists():
            oss_key = f"csc-training/{f}"
            print(f"Uploading: {f} → oss://{oss_key}")
            bucket.put_object_from_file(oss_key, f)

    print(f"\nUploaded {count} data files to OSS")


def submit_job(args):
    """Submit a PAI-DLC training job."""
    from alibabacloud_pai_dlc20201203.client import Client
    from alibabacloud_pai_dlc20201203.models import CreateJobRequest
    from alibabacloud_tea_openapi.models import Config

    key_id = os.environ["ALIBABA_ACCESS_KEY_ID"]
    key_secret = os.environ["ALIBABA_ACCESS_KEY_SECRET"]
    region = os.environ["ALIBABA_REGION"]
    bucket_name = os.environ["ALIBABA_OSS_BUCKET"]

    config = Config(
        access_key_id=key_id,
        access_key_secret=key_secret,
        region_id=region,
        endpoint=f"pai-dlc.{region}.aliyuncs.com",
    )
    client = Client(config)

    oss_data_uri = f"oss://{bucket_name}/csc-training/"
    oss_output_uri = f"oss://{bucket_name}/csc-training/output/"

    # Training command
    train_cmd = (
        "pip install -r /mnt/data/requirements.txt && "
        "python /mnt/data/scripts/train.py "
        "--config /mnt/data/config/training_config.yaml "
        "--data-dir /mnt/data/datasets"
    )

    request = CreateJobRequest(
        display_name=f"csc-finetune-v{args.version}",
        job_type="TFJob",
        job_specs=[
            {
                "type": "Worker",
                "image": "registry.cn-shanghai.aliyuncs.com/pai-dlc/pytorch-training:2.1-gpu-py310-cu121-ubuntu22.04",
                "pod_count": 1,
                "ecs_spec": "ecs.gn6i-c4g1.xlarge",  # 1x T4 16GB
                "resource": {"gpu": 1, "cpu": 4, "memory": "16Gi"},
            }
        ],
        data_sources=[
            {
                "data_source_type": "OSS",
                "uri": oss_data_uri,
                "mount_path": "/mnt/data",
            }
        ],
        code_source={"mount_path": "/mnt/code"},
        user_command=train_cmd,
        max_running_time_minutes=120,
    )

    resp = client.create_job(request)
    job_id = resp.body.job_id
    print(job_id)
    return job_id


def wait_for_job(args):
    """Wait for a PAI-DLC job to complete."""
    from alibabacloud_pai_dlc20201203.client import Client
    from alibabacloud_tea_openapi.models import Config

    key_id = os.environ["ALIBABA_ACCESS_KEY_ID"]
    key_secret = os.environ["ALIBABA_ACCESS_KEY_SECRET"]
    region = os.environ["ALIBABA_REGION"]

    config = Config(
        access_key_id=key_id,
        access_key_secret=key_secret,
        region_id=region,
        endpoint=f"pai-dlc.{region}.aliyuncs.com",
    )
    client = Client(config)

    job_id = args.job_id
    print(f"Waiting for job: {job_id}")

    terminal_states = {"Succeeded", "Failed", "Stopped"}
    poll_interval = 30  # seconds

    while True:
        resp = client.get_job(job_id)
        status = resp.body.status
        print(f"  Status: {status}")

        if status in terminal_states:
            if status != "Succeeded":
                print(f"Job {status}!")
                sys.exit(1)
            print("Job succeeded!")
            return

        time.sleep(poll_interval)


def download_model(args):
    """Download trained model from OSS."""
    bucket = get_oss_bucket()
    bucket_name = os.environ["ALIBABA_OSS_BUCKET"]
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    oss_prefix = "csc-training/output/finetuned/merged/"

    # List files under the merged model output
    for obj in oss2.ObjectIteratorV2(bucket, prefix=oss_prefix):
        if obj.key.endswith("/"):
            continue
        filename = obj.key[len(oss_prefix) :]
        local_path = output_dir / filename
        local_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading: oss://{obj.key} → {local_path}")
        bucket.get_object_to_file(obj.key, str(local_path))

    print(f"\nModel downloaded to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="PAI-DLC training helper")
    sub = parser.add_subparsers(dest="command", required=True)

    # upload-data
    p_upload = sub.add_parser("upload-data", help="Upload datasets to OSS")
    p_upload.add_argument("--data-dir", default="datasets")

    # submit
    p_submit = sub.add_parser("submit", help="Submit PAI-DLC training job")
    p_submit.add_argument("--config", default="config/training_config.yaml")
    p_submit.add_argument("--version", required=True)

    # wait
    p_wait = sub.add_parser("wait", help="Wait for training job")
    p_wait.add_argument("--job-id", required=True)

    # download-model
    p_download = sub.add_parser("download-model", help="Download trained model from OSS")
    p_download.add_argument("--output", default="output/finetuned/merged")

    args = parser.parse_args()

    if args.command == "upload-data":
        upload_data(args)
    elif args.command == "submit":
        submit_job(args)
    elif args.command == "wait":
        wait_for_job(args)
    elif args.command == "download-model":
        download_model(args)


if __name__ == "__main__":
    main()
