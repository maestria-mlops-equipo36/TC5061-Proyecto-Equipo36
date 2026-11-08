# Imagen base ligera compatible con arquitecturas ARM64 (Mac Apple Silicon) y x86_64
FROM python:3.12-slim

# Evitar escritura de bytecode, habilitar logs sin búfer y configurar PYTHONPATH
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000

WORKDIR /app

# Instalar utilerías esenciales para red y compilación
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar manifiesto e instalar librerías
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fuente, módulos de serving y documentación
COPY mibici/ /app/mibici/
COPY README.md /app/
# Crear usuario y grupo sin privilegios
RUN groupadd --system appgroup && \
    useradd --system --gid appgroup --home-dir /app appuser && \
    chown -R appuser:appgroup /app

# Ejecutar la aplicación sin privilegios de root
USER appuser

# Exponer el puerto del servicio REST
EXPOSE 8000

# Endpoint de monitoreo de salud integrado para el motor de contenedores
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Comando de arranque con Uvicorn
CMD ["python", "-m", "uvicorn", "mibici.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
