FROM continuumio/miniconda3:latest

WORKDIR /app

COPY environment.yml /app/environment.yml

RUN conda env create -f environment.yml && conda clean -afy

SHELL ["conda", "run", "-n", "atr-discovery-studio", "/bin/bash", "-c"]

COPY . /app

EXPOSE 7860

CMD ["conda", "run", "--no-capture-output", "-n", "atr-discovery-studio", "streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
