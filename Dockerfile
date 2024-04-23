FROM python:3.9-slim-buster
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
RUN pip install xgboost
CMD ["streamlit", "run", "app.py"]