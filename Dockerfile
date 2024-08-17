# Use uma imagem base do Python
FROM python:3.10-slim

# Defina o diretório de trabalho
WORKDIR /app

# Copie os arquivos de requisitos para o contêiner
COPY requirements.txt .

# Instale as dependências
RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN pip install -r requirements.txt

# Copie o restante dos arquivos para o contêiner
COPY . .

# Exponha a porta na qual a aplicação Flask será executada
EXPOSE 5000

# Comando para executar a aplicação
CMD ["python", "app.py"]
