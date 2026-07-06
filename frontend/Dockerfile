FROM python:3.11-slim

WORKDIR /app

# Install streamlit and requests only — frontend doesn't need ML libraries
RUN pip install streamlit==1.57.0 requests --no-cache-dir

# Copy only the frontend app
COPY frontend/app.py .

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]