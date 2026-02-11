FROM python:3.12-slim

# Éviter que Python ne génère des fichiers .pyc et forcer l'affichage des logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1


WORKDIR /app


COPY requirements.txt .

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt


COPY . .


CMD ["sh", "-c", "find ml nlp-pipeline -name '*.py' ! -name '__*' -exec python {} \\;"]