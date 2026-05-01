# ==============================================================
# DOCKERFILE - Pipeline ML
# ==============================================================
# Ce fichier permet de conteneuriser le pipeline ML.
# Pour construire : docker build -t ecommerce-pipeline -f pipeline/Dockerfile .
# Pour lancer     : docker run ecommerce-pipeline
# ==============================================================

FROM python:3.10-slim

# Repertoire de travail
WORKDIR /app

# Copier les fichiers de dependances
COPY requirements.txt .

# Installer les dependances
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code
COPY . .

# Lancer le pipeline
CMD ["python", "pipeline/kubeflow_pipeline.py"]
