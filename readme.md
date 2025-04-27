# CS6460-OpenMind

## Project Introduction

The CS6460-OpenMind project aims to build a cognitive enhancement learning system based on an open-source ecosystem and multimodal interaction. Addressing the pain points of existing intelligent learning tools, such as single interaction modes, closed-source architectures, and static knowledge output, this project integrates multiple modalities like speech, text, and vision. It achieves structured, dynamic knowledge presentation and multi-sensory cognitive support, thereby enhancing learners' depth of understanding and memory retention.

The system uses Python as the main development language, with the interface implemented using CustomTkinter. Core functionalities include:

*   **Multimodal Input:** Supports speech recognition (based on the Wav2Vec2 model) and text input.
*   **Dynamic Knowledge Generation:** Calls backend AI services (based on the Qwen large language model) to generate structured mind maps and detailed answers.
*   **Visual Enhancement:** Utilizes open-source image generation models (e.g., Janus Pro 7B model) to automatically generate relevant educational images, and provides an image viewer with zoom and pan support.
*   **Auditory Support:** Splits the AI response text into sentences and uses TTS (Text-to-Speech) functionality for sentence-by-sentence playback, providing a multi-sensory learning experience.
*   **Online Search Integration:** Optionally integrates Google search functionality to provide the AI with the latest web information, improving the timeliness and accuracy of responses.
*   **Local Deployment:** Supports complete local deployment, ensuring user data privacy.

By combining mind maps, educational images, text, and speech, the OpenMind system provides learners with a more immersive and efficient intelligent learning environment.

## Installation Guide

This project consists of two parts, frontend and backend, which need to be installed and configured separately.

**Note:** The backend part depends on PyTorch and potentially GPU acceleration. Please ensure your environment meets the relevant requirements. Additionally, the backend depends on the Ollama service, which needs to be installed and the Qwen model pulled beforehand (`ollama pull qwen`). The frontend's mind map generation depends on `markmap-cli` and `Playwright`.

### Prerequisites

*   Python 3.8+
*   (Recommended) NVIDIA GPU and corresponding CUDA environment (for backend and some frontend model acceleration)
*   Ollama (needs to be installed and running, and the Qwen model pulled, e.g., `ollama pull qwen`)
*   Node.js and npm or yarn (for installing markmap-cli)
*   Playwright (for mind map screenshots)
*   **Janus Pro (for image generation)**

### Steps

1.  **Install Backend Dependencies:**
    *   Navigate to the project directory.
    *   **Navigate to the backend directory (if the backend code is in a subdirectory)**
    *   Install Python dependencies. Please note that the following PyTorch installation command is for CUDA 12.1; adjust the `--index-url` according to your CUDA version.
        ```bash
        # Ensure you are in the backend directory
        pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
        pip install numpy pandas tts fastapi pillow transformers uvicorn huggingface_hub ollama
        ```
    *   **Install Janus Pro (for Image Generation Functionality):**
        *   Clone the Janus repository to a suitable location (e.g., into a subdirectory within the backend directory, or alongside the backend directory). The instructions below assume you are cloning it into the backend directory.
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
        *   **Return to the backend directory (if you need to proceed with subsequent steps):**
            ```bash
            cd .. # Or adjust based on your actual directory structure
            ```
    *   Ensure Ollama is running and the Qwen model has been pulled (`ollama pull qwen`).
    *   Start the backend service (refer to the backend documentation or code for the specific startup command). It might typically be similar to:
        ```bash
        python backend.py
        ```

2.  **Install Frontend Dependencies:**
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

After completing the above steps, you should be able to start and use the OpenMind system. Please ensure you start the backend service first before starting the frontend application.
