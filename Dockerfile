# ==============================================================
# DOCKERFILE PRINCIPAL - Dashboard Streamlit
# ==============================================================
# Pour construire : docker build -t ecommerce-dashboard .
# Pour lancer     : docker run -p 8501:8501 ecommerce-dashboard
# ==============================================================

FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
