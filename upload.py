from huggingface_hub import HfApi
from loguru import logger


def main():
    api = HfApi()

    path = "save/my_model/pytorch_model.bin"

    logger.info("Uploading model to huggingface hub")
    api.upload_file(
        path_or_fileobj=path,
        path_in_repo="pytorch_model.bin",
        repo_id="Bingsu/temp_vilt_vqa",
        repo_type="model",
    )


if __name__ == "__main__":
    main()
