FROM ultralytics/ultralytics:8.2.25-python

RUN pip install poetry

WORKDIR /opt

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

WORKDIR /opt

# COPY pyproject.toml poetry.lock ./

# RUN poetry config virtualenvs.create false && poetry install
COPY req.txt req.txt
RUN pip install --no-cache-dir -r req.txt

COPY . ./

CMD python -m app.main