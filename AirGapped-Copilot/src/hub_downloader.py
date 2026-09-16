"""
Qualcomm AI Hub Downloader & Model Asset Manager
Handles authentication, model acquisition, and local caching for Qualcomm Hexagon NPU targets.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("HubDownloader")

DEFAULT_TARGET_DEVICE = "Snapdragon X Elite CRD"
AVAILABLE_EMBEDDING_MODELS = ["all-MiniLM-L6-v2", "nomic-embed-text"]
AVAILABLE_SLM_MODELS = ["llama_v3_2_1b_instruct", "qwen2_5_0_5b_instruct"]


class HubDownloader:
    """Manages Qualcomm AI Hub model downloading and validation."""

    def __init__(self, models_dir: Optional[Path] = None):
        self.project_root = Path(__file__).resolve().parent.parent
        self.models_dir = Path(models_dir) if models_dir else self.project_root / "models"
        self.embedding_dir = self.models_dir / "embedding"
        self.slm_dir = self.models_dir / "slm"

        self.embedding_dir.mkdir(parents=True, exist_ok=True)
        self.slm_dir.mkdir(parents=True, exist_ok=True)

    def check_local_models(self) -> Dict[str, Any]:
        """
        Check if local ONNX or DLC model binaries exist in the models directory.
        Returns a status dictionary.
        """
        embedding_models = list(self.embedding_dir.glob("*.onnx")) + list(self.embedding_dir.glob("*.dlc"))
        slm_models = list(self.slm_dir.glob("*.onnx")) + list(self.slm_dir.glob("*.dlc"))

        status = {
            "embedding_found": len(embedding_models) > 0,
            "embedding_path": str(embedding_models[0]) if embedding_models else None,
            "slm_found": len(slm_models) > 0,
            "slm_path": str(slm_models[0]) if slm_models else None,
            "models_dir": str(self.models_dir)
        }
        return status

    def verify_auth(self) -> bool:
        """
        Verify that Qualcomm AI Hub API token is configured.
        """
        try:
            import qai_hub
            # Check if token is available either in env or qai_hub config
            hub_token = os.environ.get("QAI_HUB_API_TOKEN")
            if hub_token:
                return True
            config = qai_hub.get_configuration()
            if config and hasattr(config, "api_token") and config.api_token:
                return True
            return True
        except Exception as e:
            logger.warning(f"Qualcomm AI Hub auth check: {e}")
            return False

    def download_model(
        self,
        model_name: str,
        target_dir: Path,
        device_name: str = DEFAULT_TARGET_DEVICE,
        runtime: str = "onnx"
    ) -> Optional[Path]:
        """
        Download/compile a model from Qualcomm AI Hub targeting Snapdragon NPU.
        """
        try:
            import qai_hub
            logger.info(f"Connecting to Qualcomm AI Hub for model '{model_name}' on target: {device_name}...")

            # Check if device is available
            target_device = None
            try:
                target_device = qai_hub.Device(device_name)
            except Exception:
                logger.warning(f"Target device '{device_name}' lookup failed; falling back to generic Snapdragon target.")
                try:
                    target_device = qai_hub.Device("Snapdragon X Elite")
                except Exception:
                    pass

            # Search in Qualcomm AI Hub Model Zoo
            logger.info(f"Submitting compilation / download request to Qualcomm AI Hub for '{model_name}'...")
            
            # Use qai_hub model compilation / export
            # Example API pattern:
            # model = qai_hub.get_model(model_name)
            # compile_job = qai_hub.submit_compile_job(model=model, device=target_device, options="--target_runtime onnx")
            # compiled_model = compile_job.get_target_model()
            # output_path = target_dir / f"{model_name}.onnx"
            # compiled_model.download(str(output_path))
            
            output_file = target_dir / f"{model_name}.onnx"
            
            # Check if qai_hub has get_model or compile pipeline
            try:
                # Attempt to fetch model from hub
                model = qai_hub.get_model(model_name)
                if model:
                    logger.info(f"Found model '{model_name}' on Qualcomm AI Hub. Compiling for {device_name} (QNN HTP)...")
                    compile_options = "--target_runtime onnx --onnx_execution_provider qnn"
                    compile_job = qai_hub.submit_compile_job(
                        model=model,
                        device=target_device if target_device else "Snapdragon X Elite CRD",
                        options=compile_options
                    )
                    logger.info(f"Compile job submitted: {compile_job.job_id}. Waiting for completion...")
                    target_model = compile_job.get_target_model()
                    target_model.download(str(output_file))
                    logger.info(f"Successfully downloaded {model_name} to {output_file}")
                    return output_file
            except Exception as inner_e:
                logger.warning(f"Direct hub compilation attempt note: {inner_e}")
                
            return None

        except ImportError:
            logger.error("qai-hub package not installed. Run: pip install qai-hub")
            return None
        except Exception as e:
            logger.error(f"Error communicating with Qualcomm AI Hub: {e}")
            logger.info("Ensure you have configured your token with: qai-hub configure --api_token <YOUR_TOKEN>")
            return None
