FROM python:3.13-slim-bookworm
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY inference.proto ./
RUN python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. inference.proto
COPY service.py model.json ./
USER 65534:65534
EXPOSE 8080 50051
CMD ["python", "service.py"]
