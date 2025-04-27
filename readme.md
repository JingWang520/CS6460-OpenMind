# CS6460-OpenMind

## Project Introduction

The CS6460-OpenMind project aims to build a cognitive enhancement learning system based on an open-source ecosystem and multi-modal interaction. Addressing the pain points of existing intelligent learning tools, such as single interaction modes, closed-source architecture, and static knowledge output, this project integrates multiple modalities including voice, text, and vision. It aims to achieve structured, dynamic knowledge presentation and multi-sensory cognitive support, thereby enhancing learners' understanding depth and memory retention.

The system uses Python as the main development language, with the interface implemented using CustomTkinter. Core functionalities include:

*   **Multi-modal Input:** Supports voice recognition (based on Wav2Vec2 model) and text input.
*   **Dynamic Knowledge Generation:** Calls backend AI services (based on Qwen large language model) to generate structured mind maps and detailed answers.
*   **Visual Enhancement:** Utilizes open-source image generation models (such as Janus Pro 7B model) to automatically generate relevant educational images, and provides an image viewer supporting zoom and pan.
*   **Auditory Support:** Splits AI answer text into sentences and uses TTS (Text-to-Speech) functionality to play them sentence by sentence, providing a multi-sensory learning experience.
*   **Online Search Integration:** Optionally integrates Google search functionality to provide the AI with the latest web information, improving answer timeliness and accuracy.
*   **Local Deployment:** Supports full local deployment, ensuring user data privacy.

By combining mind maps, educational images, text, and voice, the OpenMind system provides learners with a more immersive and efficient intelligent learning environment.

## Installation Guide

This project consists of two parts: frontend and backend, which need to be installed and configured separately.

**Note:** The backend part relies on PyTorch and potential GPU acceleration. Please ensure your environment meets the relevant requirements. Additionally, the backend depends on the Ollama service, which needs to be installed and the Qwen model pulled beforehand. The frontend's mind map generation relies on `markmap-cli` and `Playwright`.

### Prerequisites

*   Python 3.8+
*   (Recommended) Two NVIDIA GPUs 3090 or higher and corresponding CUDA environment (for backend and partial frontend model acceleration).
*   Ollama (needs to be installed and running, and the Qwen model pulled, e.g., `ollama pull qwen`).
*   Node.js and npm or yarn (for installing markmap-cli).
*   Playwright (for mind map screenshots).
*   **Janus Pro (for image generation)**

### Steps

1.  **Backend Dependency Installation:**
    *   Navigate to the project directory.
    *   **Navigate to the backend directory (if the backend code is in a subdirectory)**
    *   Install Python dependencies. Note that the following PyTorch installation command is for CUDA 12.1; please adjust the `--index-url` according to your CUDA version.
        ```bash
        # Ensure you are in the backend directory
        pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
        pip install numpy pandas tts fastapi pillow transformers uvicorn huggingface_hub ollama
        ```
    *   **Install Janus Pro (for image generation functionality):**
        *   Clone the Janus repository to a suitable location (e.g., a subdirectory within the backend directory, or parallel to the backend directory). This assumes you are cloning from within the backend directory.
            ```bash
            git clone https://github.com/deepseek-ai/Janus.git
            ```
        *   Navigate into the cloned Janus directory.
            ```bash
            cd Janus
            ```
        *   Install Janus Pro. The `-e` parameter means installing in editable mode.
            ```bash
            pip install -e .
            ```
        *   **Return to the backend directory (if you need to continue with subsequent steps):**
            ```bash
            cd .. # or adjust based on your actual directory structure
            ```
    *   Ensure Ollama is running and you have pulled the Qwen model (`ollama pull qwen`):
        ```bash
        curl -fsSL https://ollama.com/install.sh | sh
        ollama pull qwen
        ollama serve # Ensure Ollama server is running
        ```
    *   Start the backend service:
        ```bash
        python backend.py
        ```

2.  **Frontend Dependency Installation:**
    *   Navigate to the project directory.
    *   Install Python dependencies.
        ```bash
        pip install sounddevice numpy scipy customtkinter Pillow torch transformers googlesearch-python requests readability-lxml
        ```
    *   Install `markmap-cli` (requires Node.js environment).
        ```bash
        npm install -g markmap-cli
        # Or using yarn
        # yarn global add markmap-cli
        ```
    *   Install Playwright browser binaries.
        ```bash
        playwright install
        ```
    *   Run the frontend application:
        ```bash
        python main_GUI.py
        ```

After completing the above steps, you should be able to start and use the OpenMind system. Please ensure you start the backend service first, then the frontend application.
