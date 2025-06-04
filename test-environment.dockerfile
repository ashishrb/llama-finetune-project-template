FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest

# Install conda/mamba
RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O ~/miniforge.sh && \
    bash ~/miniforge.sh -b -p ~/miniforge && \
    rm ~/miniforge.sh

ENV PATH="~/miniforge/bin:$PATH"

# Test basic packages first
RUN pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
RUN pip install transformers==4.37.2 datasets==2.18.0 peft==0.8.2 accelerate==0.26.1

# Test Unsloth installation
RUN pip install packaging ninja
RUN pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" || echo "Unsloth failed"

# Test imports
RUN python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
RUN python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
RUN python -c "try: import unsloth; print('Unsloth: OK'); except: print('Unsloth: Failed')"