# Use Python 3.13 as the base image
FROM python:3.13-slim

# Install uv directly from the official astral image for maximum speed
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the working directory
WORKDIR /app

# Copy dependency files first to leverage Docker layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies using uv
# --frozen ensures it uses the lockfile, --no-cache keeps the image small
RUN uv sync --frozen --no-cache

# Copy the rest of the application code
COPY app.py init_db.py ./
COPY rag/ ./rag/

# Expose the port your FastAPI server runs on
EXPOSE 8000

# Command to run the application using the virtual environment created by uv
CMD ["/app/.venv/bin/uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
