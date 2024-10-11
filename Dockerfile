FROM python:3.11
WORKDIR /classification
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
CMD ["python", "classification.py"]