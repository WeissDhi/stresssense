# Gunakan image Python yang ringan
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Salin requirements dan install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua file aplikasi ke dalam container
COPY . .

# Railway menggunakan PORT environment variable
ENV PORT=5000

# Expose port
EXPOSE 5000

# Jalankan aplikasi Flask
CMD ["python", "app.py"]