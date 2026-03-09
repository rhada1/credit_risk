# ════════════════════════════════════════════
# Dockerfile — API Prédiction de Risque de Crédit
# ════════════════════════════════════════════

# Image de base Python légère
FROM python:3.13-slim

# Répertoire de travail dans le conteneur
WORKDIR /app

# Copier requirements en premier (optimise le cache Docker)
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste du projet
COPY src/ ./src/
COPY models/ ./models/

# Port exposé par l'API
EXPOSE 8000

# Commande de démarrage
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]