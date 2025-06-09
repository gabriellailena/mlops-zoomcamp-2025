FROM python:3.10-slim

RUN pip install mlflow==2.22.0

RUN mkdir -p /home/mlflow_data/mlruns /home/mlflow_data/artifacts && \
    chmod -R 777 /home/mlflow_data

# Set working directory
WORKDIR /home/mlflow_data

# Expose port
EXPOSE 5000

# Default command
CMD [ \
    "mlflow", "server", \
    "--backend-store-uri", "file:///home/mlflow_data/mlruns", \
    "--default-artifact-root", "file:///home/mlflow_data/artifacts", \
    "--host", "0.0.0.0", \
    "--port", "5000" \
]