FROM python:3.9-slim

WORKDIR /app

RUN pip install --no-cache-dir jupyterlab pandas numpy matplotlib scikit-learn seaborn

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"] 