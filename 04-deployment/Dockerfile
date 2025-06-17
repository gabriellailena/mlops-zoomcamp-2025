FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

# Install dependencies
RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./"]

# Install dependencies directly into the system Python environment
RUN pipenv install --system --deploy

# Copy script
COPY ["predict.py", "./"]
ENTRYPOINT ["python", "predict.py"]
CMD ["--year", "2023", "--month", "5"]