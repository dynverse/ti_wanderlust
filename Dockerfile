FROM dynverse/dynwrappy:v0.1.0

RUN pip install git+https://github.com/dynverse/pywishbone --upgrade --upgrade-strategy only-if-needed

COPY definition.yml run.py example.sh /code/

ENTRYPOINT ["/code/run.py"]
