FROM dynverse/dynwrappy:v0.1.0

RUN pip install git+https://github.com/dynverse/pywishbone --upgrade --upgrade-strategy only-if-needed

COPY definition.yml example.h5 run.py /code/

ENTRYPOINT ["/code/run.py"]
