FROM python:3.8-slim-buster AS production

ENV PYTHONUNBUFFERED=1
WORKDIR /app/

RUN apt-get update && \
    apt-get install -y bash build-essential gcc libffi-dev musl-dev \
      openssl postgresql libpq-dev

COPY requirements/prod.txt ./requirements/prod.txt
RUN python -m pip install --no-cache-dir --upgrade pip==22.0.4
RUN pip install -r ./requirements/prod.txt

COPY manage.py ./manage.py
# COPY setup.cfg ./setup.cfg
COPY bstl_website ./bstl_website

EXPOSE 8000
