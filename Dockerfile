FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

WORKDIR /code
ENV PYTHONPATH "${PYTHONPATH}:/code"
COPY . /code/
RUN chmod +x utils/*.sh

