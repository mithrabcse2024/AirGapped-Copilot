#!/usr/bin/env python3
"""
Qualcomm AI Hub Model Downloader Script
Targets Snapdragon X Elite / X Plus / Snapdragon 8 Gen series Hexagon NPU.
Downloads and optimizes:
  1. Text Embedding Model: `all-MiniLM-L6-v2` or `nomic-embed-text`
  2. Small Language Model: `llama_v3_2_1b_instruct` or `qwen2_5_0_5b_instruct`
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.hub_downloader import HubDownloader, DEFAULT_TARGET_DEVICE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AIHubDownloadScript")


def create_mock_models(models_dir: Path):
    """
    Creates lightweight dummy ONNX models for testing the offline execution pipeline
    when development/testing is conducted prior to acquiring or configuring Qualcomm AI Hub API keys.
    """
    logger.info("Generating mock ONNX models for testing offline pipeline...")
    try:
        import numpy as np
        import onnx
        from onnx import helper, TensorProto

        # 1. Embedding Model (Input: input_ids [batch, seq_len], Output: embeddings [batch, 384])
        emb_path = models_dir / "embedding" / "all-MiniLM-L6-v2.onnx"
        emb_path.parent.mkdir(parents=True, exist_ok=True)
        if not emb_path.exists():
            # Build a simple ONNX graph that outputs a 384-d vector
            input_ids = helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, "seq_len"])
            output = helper.make_tensor_value_info("embeddings", TensorProto.FLOAT, [1, 384])
            
            # Constant 384-d float vector
            const_tensor = helper.make_tensor("const_emb", TensorProto.FLOAT, [1, 384], [0.05] * 384)
            node = helper.make_node("Constant", inputs=[], outputs=["embeddings"], value=const_tensor)
            
            graph = helper.make_graph([node], "mock_embedding", [input_ids], [output])
            model = helper.make_model(graph, producer_name="qai_hub_mock")
            onnx.save(model, str(emb_path))
            logger.info(f"Created mock embedding ONNX model: {emb_path}")

        # 2. SLM Model (Input: input_ids [batch, seq_len], Output: logits [batch, seq_len, vocab_size])
        slm_path = models_dir / "slm" / "llama_v3_2_1b_instruct.onnx"
        slm_path.parent.mkdir(parents=True, exist_ok=True)
        if not slm_path.exists():
            input_ids = helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, "seq_len"])
            logits = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 1, 32000])
            
            const_tensor = helper.make_tensor("const_logits", TensorProto.FLOAT, [1, 1, 32000], [0.0] * 32000)
            node = helper.make_node("Constant", inputs=[], outputs=["logits"], value=const_tensor)
            
            graph = helper.make_graph([node], "mock_slm", [input_ids], [logits])
            model = helper.make_model(graph, producer_name="qai_hub_mock")
            onnx.save(model, str(slm_path))
            logger.info(f"Created mock SLM ONNX model: {slm_path}")

        logger.info("Mock models successfully generated in ./models/ for offline testing.")
    except Exception as e:
        logger.warning(f"Could not generate mock ONNX models: {e}. (Install 'onnx' if testing without AI Hub credentials)")


def main():
    parser = argparse.ArgumentParser(description="Download optimized Snapdragon NPU models from Qualcomm AI Hub")
    parser.add_argument(
        "--embedding-model",
        choices=["all-MiniLM-L6-v2", "nomic-embed-text"],
        default="all-MiniLM-L6-v2",
        help="Text embedding model to download from Qualcomm AI Hub"
    )
    parser.add_argument(
        "--slm-model",
        choices=["llama_v3_2_1b_instruct", "qwen2_5_0_5b_instruct"],
        default="llama_v3_2_1b_instruct",
        help="Small Language Model to download from Qualcomm AI Hub"
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_TARGET_DEVICE,
        help="Snapdragon target device (e.g., 'Snapdragon X Elite CRD', 'Snapdragon X Plus')"
    )
    parser.add_argument(
        "--api-token",
        default=None,
        help="Qualcomm AI Hub API Token (or set QAI_HUB_API_TOKEN env var)"
    )
    parser.add_argument(
        "--create-mock-if-missing",
        action="store_true",
        help="Generate lightweight mock ONNX assets if Qualcomm AI Hub credentials are not yet set"
    )
    args = parser.parse_args()

    models_dir = PROJECT_ROOT / "models"
    downloader = HubDownloader(models_dir=models_dir)

    print("=" * 70)
    print("  AIR-GAPPED COPILOT - QUALCOMM AI HUB MODEL DOWNLOADER")
    print("  Targeting Snapdragon NPU (Hexagon HTP / QNN Execution Provider)")
    print("=" * 70)

    # 1. Fallback check: check if models already exist locally
    status = downloader.check_local_models()
    print("\n[STEP 1] Checking existing local models...")
    print(f"Models directory: {status['models_dir']}")
    print(f"Embedding model found: {status['embedding_found']} ({status['embedding_path']})")
    print(f"SLM model found:       {status['slm_found']} ({status['slm_path']})")

    if status["embedding_found"] and status["slm_found"]:
        print("\nAll required model binaries are already present locally in ./models/.")
        print("Ready for 100% offline air-gapped execution.")
        return

    # 2. Check Qualcomm AI Hub Token
    if args.api_token:
        os.environ["QAI_HUB_API_TOKEN"] = args.api_token

    print("\n[STEP 2] Verifying Qualcomm AI Hub authentication...")
    try:
        import qai_hub
        auth_ok = downloader.verify_auth()
        if not auth_ok:
            print("\nQualcomm AI Hub credentials not detected.")
            print("Please configure your API token using:")
            print("  qai-hub configure --api_token <YOUR_QUALCOMM_AI_HUB_API_TOKEN>")
            print("or pass --api-token <KEY> or set the QAI_HUB_API_TOKEN environment variable.")
            print("Sign up / retrieve token at: https://aihub.qualcomm.com")
            
            if args.create_mock_if_missing:
                create_mock_models(models_dir)
            return
        print("Qualcomm AI Hub authentication verified.")
    except ImportError:
        print("\n[!] 'qai-hub' SDK is not installed.")
        print("Please run: pip install qai-hub")
        if args.create_mock_if_missing:
            create_mock_models(models_dir)
        return

    # 3. Download Embedding Model
    print(f"\n[STEP 3] Fetching embedding model: {args.embedding_model} for {args.device}...")
    if not status["embedding_found"]:
        emb_file = downloader.download_model(
            model_name=args.embedding_model,
            target_dir=models_dir / "embedding",
            device_name=args.device
        )
        if emb_file:
            print(f"Embedding model downloaded: {emb_file}")
        else:
            print(f"Could not automatically compile/download {args.embedding_model} via Hub API.")
    else:
        print(f"Skipping embedding download: already exists at {status['embedding_path']}")

    # 4. Download SLM Model
    print(f"\n[STEP 4] Fetching SLM model: {args.slm_model} for {args.device}...")
    if not status["slm_found"]:
        slm_file = downloader.download_model(
            model_name=args.slm_model,
            target_dir=models_dir / "slm",
            device_name=args.device
        )
        if slm_file:
            print(f"SLM model downloaded: {slm_file}")
        else:
            print(f"Could not automatically compile/download {args.slm_model} via Hub API.")
    else:
        print(f"Skipping SLM download: already exists at {status['slm_path']}")

    # Final summary
    final_status = downloader.check_local_models()
    print("\n" + "=" * 70)
    print("Download Summary:")
    print(f"  Embedding model present: {final_status['embedding_found']}")
    print(f"  SLM model present:       {final_status['slm_found']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
