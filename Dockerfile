FROM tensorflow/tensorflow:2.4.1-gpu

RUN apt-get update && \
    apt-get install -y git

RUN curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python && source $HOME/.poetry/env

COPY . /app/healthfact

WORKDIR /app/healthfact

# RUN curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash
# RUN $HOME/.pyenv/bin/pyenv install 3.8.10

RUN $HOME/.poetry/bin/poetry install
# RUN $HOME/.poetry/bin/poetry config virtualenvs.create false && $HOME/.poetry/bin/poetry install