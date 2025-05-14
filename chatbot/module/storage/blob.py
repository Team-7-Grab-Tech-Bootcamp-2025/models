import os
import requests
from uuid import uuid4

from azure.storage.blob import BlobServiceClient

from src.config.constant import AzureBlobStorageCFG, CoreCFG
from src.utils.logger import logger

blob_service_client = BlobServiceClient.from_connection_string(
    AzureBlobStorageCFG.AZURE_BLOB_CONNECTION_STRING
)
container_client = blob_service_client.get_container_client(
    AzureBlobStorageCFG.CONTAINER_NAME
)


def upload_file_to_blob(file_path: str, execution_id: str = str(uuid4())):
    blob_name = os.path.join(f"iqgm_{execution_id}/", os.path.basename(file_path))
    blob_client = container_client.get_blob_client(blob_name)
    try:
        with open(file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
        logger.info(f"File {file_path} uploaded to blob {blob_name}.")

        # Get the URL of the uploaded blob
        blob_url = blob_client.url
        logger.info(f"Blob URL: {blob_url}")
        return blob_url
    except Exception as e:
        logger.error(f"Failed to upload file: {e}")
        return False


def upload_file_to_blob_from_url(image_url: str, image_file_name: str = "generated"):
    try:
        generated_image = requests.get(image_url).content
        blob_name = os.path.join(f"iqgm_{uuid4()}/", f"{image_file_name}.jpg")
        blob_client = container_client.get_blob_client(blob_name)

        blob_client.upload_blob(generated_image, overwrite=True)
        logger.info(f"File {image_url} uploaded to blob {blob_name}.")

        # Get the URL of the uploaded blob
        blob_url = blob_client.url
        logger.info(f"Blob URL: {blob_url}")
        return blob_url
    except Exception as e:
        logger.error(f"Failed to upload file: {e}")
        return False


def download_file(file_url: str, conversation_id: str):
    try:
        file_name = file_url.split("/")[-1]
        execution_dir = os.path.join(CoreCFG.OUTPUT_DIR, conversation_id)
        logger.info(execution_dir)
        os.makedirs(execution_dir, exist_ok=True)
        folder_path = f"{CoreCFG.SYSTEM_OS_PATH}{execution_dir}"
        file_path = folder_path + "/" + file_name

        response = requests.get(file_url)

        with open(file_path, mode="wb") as file:
            file.write(response.content)

        logger.info(f"File {file_path} downloaded successfully.")

        return file_path

    except Exception as e:
        logger.error(f"Failed to upload file: {e}")
        return "file_path"
