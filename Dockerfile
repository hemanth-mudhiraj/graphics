FROM python:3.11-slim

WORKDIR /app

# Install CPU-only PyTorch (smaller image for deployment)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY best_model.pth ./best_model.pth

ENV MODEL_PATH=/app/best_model.pth
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
