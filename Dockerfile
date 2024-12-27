FROM continuumio/miniconda3 AS builder

RUN /opt/conda/bin/conda init bash && \
    /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda create -n fastapi python=3.11 -y && \
    conda activate fastapi && \
    pip install 'fastapi[standard]' && \
    pip install docker && \
    git clone https://github.com/luckycontrol/MOAI-server.git && \
    conda clean -afy"

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir /app && \
    cp -r /MOAI-server /app/MOAI-server && \
    cp -r /opt/conda /app/conda

FROM continuumio/miniconda3

ENV PATH=/opt/conda/bin:$PATH

COPY --from=builder /app/MOAI-server /MOAI-server
COPY --from=builder /app/conda /opt/conda

WORKDIR /MOAI-server

SHELL ["/bin/bash", "-c"]
RUN echo "source /opt/conda/etc/profile.d/conda.sh && conda activate fastapi" >> /root/.bashrc

CMD ["/bin/bash"]

VOLUME ["/moai"]