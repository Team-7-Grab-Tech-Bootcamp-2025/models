import os
import glob
import subprocess
from uuid import uuid4

from src.utils.logger import logger
from src.config.constant import CoreCFG
from src.module.storage.blob import upload_file_to_blob


def execute(code_block: str, execution_id: str = str(uuid4()), input_files: dict = {}):
    execution_dir = os.path.join(CoreCFG.OUTPUT_DIR, execution_id)
    os.makedirs(execution_dir, exist_ok=True)
    folder_path = f"{CoreCFG.SYSTEM_OS_PATH}{execution_dir}"

    try:
        result = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{folder_path}:/app",
                f"{CoreCFG.SANDBOX_IMAGE}",
                "python3",
                "-c",
                code_block,
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        logger.info(result.stdout.decode())

        files = glob.glob(os.path.join(execution_dir, "*"))

        file_results = {}
        for file_path in files:
            if os.path.basename(file_path) not in input_files:
                url = upload_file_to_blob(file_path, execution_id)
                file_results[os.path.basename(file_path)] = url

    except subprocess.CalledProcessError as e:
        logger.error("Error:", e.stderr.decode())
        return {"stdout": "Error", "stderr": e.stderr.decode(), "files": "None"}
    return {"stdout": result.stdout, "stderr": result.stderr, "files": file_results}
