FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

WORKDIR /app

COPY starter.py .
RUN pip install pandas 
RUN pip install pyarrow
RUN pip install argparse
RUN pip install scikit-learn

ENTRYPOINT ["python", "starter.py"]