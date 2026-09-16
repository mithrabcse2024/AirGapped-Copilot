
**AirGapped Copilot** is a zero-leakage, fully local retrieval-augmented generation (RAG) assistant specifically built for Snapdragon-powered HP PCs (Snapdragon X Elite, Snapdragon X Plus, and Snapdragon 8 Gen series). Designed for strictly regulated industries such as healthcare, defense, and law, AirGapped Copilot operates **100% offline**:

- **Qualcomm AI Hub Models Only**: Uses pre-optimized models directly from [Qualcomm AI Hub](https://aihub.qualcomm.com) (`all-MiniLM-L6-v2` / `nomic-embed-text` and `llama_v3_2_1b_instruct` / `qwen2_5_0_5b_instruct`).
- **Hardware Acceleration**: Executes on the **Qualcomm Hexagon NPU** via ONNX Runtime with Qualcomm's **QNN Execution Provider** (`QnnHtp.dll`) in `burst` performance mode.
- **Embedded Vector Database**: Local on-disk vector storage (LanceDB) with zero server dependencies.
- **Audit-Ready Citations**: Extracts and grounds answers directly in source PDF pages with exact page-level traceability and live NPU latency telemetry.

---

## 🏗️ Architecture Diagram

<img width="347" height="672" alt="image" src="https://github.com/user-attachments/assets/3b0e13a8-78b3-4025-90d1-374352366690" />


## 📁 Repository Structure

```
AirGapped-Copilot/
├── scripts/
│   └── download_ai_hub_models.py # Compiles & downloads ONNX/QNN binaries from Qualcomm AI Hub
├── models/                       # Local directory for Qualcomm AI Hub models (.onnx / .dlc)
│   ├── embedding/                # Optimized embedding model
│   └── slm/                      # Optimized small language model
├── data/                         # Local storage & LanceDB vector database
│   ├── lancedb/                  # Local vector database files
│   └── sample_medical_legal_brief.txt # Sample confidential brief for testing
├── src/
│   ├── __init__.py
│   ├── hub_downloader.py         # Automates qai-hub API interactions & model verification
│   ├── npu_engine.py             # ONNX Runtime QNN EP runner targeting QnnHtp.dll
│   ├── document_parser.py       # PDF parsing & chunking with page-level tracking
│   ├── vector_store.py          # Embedded local vector store (LanceDB / numpy fallback)
│   ├── rag_pipeline.py          # Offline RAG orchestration & prompt synthesizer
│   └── app_gui.py               # Dark-themed PyQt6 Native Desktop GUI
├── requirements.txt              # Complete Python dependencies
└── README.md                     # Documentation & setup guide
```

---

## 🚀 Getting Started

### 1. Prerequisites & Environment Setup

Clone this repository and create a Python virtual environment:

```bash
# Clone the repository
git clone https://github.com/your-username/AirGapped-Copilot.git
cd AirGapped-Copilot

# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Install required dependencies
pip install -r requirements.txt
```

### 2. Configure Qualcomm AI Hub Credentials

Obtain your API key from [Qualcomm AI Hub](https://aihub.qualcomm.com) and configure it:

```bash
qai-hub configure --api_token <YOUR_QUALCOMM_AI_HUB_API_KEY>
```

*(Alternatively, set the environment variable: `$env:QAI_HUB_API_TOKEN="<YOUR_TOKEN>"`)*

### 3. Download & Compile Optimized Models for Snapdragon NPU

Run the automated download script to fetch and compile the pre-optimized models targeting Snapdragon X Elite / X Plus / Snapdragon 8 Gen Hexagon NPU:

```bash
python scripts/download_ai_hub_models.py --device "Snapdragon X Elite CRD"
```

Options:
- `--embedding-model`: Choose `all-MiniLM-L6-v2` or `nomic-embed-text`
- `--slm-model`: Choose `llama_v3_2_1b_instruct` or `qwen2_5_0_5b_instruct`
- `--device`: Target Snapdragon device profile
- `--create-mock-if-missing`: Creates mock weights for offline testing before credentials are configured.

### 4. Launch the Native Desktop GUI

Run the dark-mode PyQt6 application:

```bash
python src/app_gui.py
```

---

## 💻 Hardware & NPU Execution Details

The NPU Execution Engine (`src/npu_engine.py`) initializes ONNX Runtime with Qualcomm's official **QNN Execution Provider**:

```python
import onnxruntime as ort

qnn_options = {
    "backend_path": "QnnHtp.dll",  # Qualcomm Hexagon Tensor Processor (HTP) backend
    "htp_performance_mode": "burst",  # Maximum frequency / lowest latency mode
    "htp_graph_finalization_optimization_mode": "3"
}

session = ort.InferenceSession(
    "models/embedding/all-MiniLM-L6-v2.onnx",
    providers=[("QNNExecutionProvider", qnn_options), "CPUExecutionProvider"]
)
```

- **Execution Telemetry**: Every query measures inference latency in milliseconds and reports it in the GUI header and status bar.
- **Development Fallback**: When running on development machines without Snapdragon NPU hardware, the engine automatically falls back to CPU execution with an indicator badge, ensuring seamless cross-environment testing.

---

## 🔒 Privacy & Offline Compliance

- **Zero Cloud Network Calls**: Once model assets are downloaded into `./models/`, the application operates with networking completely disconnected.
- **Local Vectors**: All chunk embeddings are stored on the local filesystem (`./data/lancedb/`).
- **HIPAA / Legal Privilege Compliance**: Safe for patient identifiable records (PHI) and privileged litigation documents.

---

## 🏆 Snapdragon AI Lab Challenge Compliance

| Requirement | Implementation |
|---|---|
| **Qualcomm AI Hub Only** | All models compiled via `qai-hub` SDK (`scripts/download_ai_hub_models.py`, `src/hub_downloader.py`). |
| **Hexagon NPU Execution** | ONNX Runtime `QNNExecutionProvider` targeting `QnnHtp.dll` in `burst` mode. |
| **100% Offline RAG** | Fully local document ingestion, vector storage, and generation. |
| **Desktop Experience** | Responsive dark-mode PyQt6 GUI with drag-and-drop dropzone, chat, and citation inspector. |
