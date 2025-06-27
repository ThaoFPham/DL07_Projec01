FROM python:3.10

WORKDIR /app

# ✅ Cài gomp để fix lỗi lightgbm
RUN apt-get update && apt-get install -y libgomp1

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["streamlit", "run", "streamlit_project01.py", "--server.port=10000", "--server.enableCORS=false"]
