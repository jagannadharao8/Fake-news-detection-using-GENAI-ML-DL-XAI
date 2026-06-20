# Use official Python runtime as a parent image
FROM python:3.11-slim

# Create a non-root user for Hugging Face Spaces security requirements
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy requirements file first to leverage Docker cache
COPY --chown=user requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY --chown=user . .

# Download the massive ML models (LFS files) directly from the Hugging Face Space
RUN rm -rf /app/artifacts/* && python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='jagannadharao8/fake-news-detection', repo_type='space', local_dir='/app', allow_patterns=['artifacts/**'])"

# Expose port 7860 (Hugging Face default)
EXPOSE 7860

# Command to run the FastAPI application
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
