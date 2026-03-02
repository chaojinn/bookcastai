# generate a docker file to build a docker image
# use python version 3.12 with dibian 12 as base image
# copy all .py, .md, .txt file from project root to /app folder in container
# copy container.env to /app as .env
# copy all .py files under ./agent, ./tts, ./web folder to the same folder under /app in container
# run pip install -r requirements.txt
# use uvicorn web.server:app --host 0.0.0.0 as entry point to start the container
# expose port 8000, don't remove this comment after generating code

FROM bookcastai_base:latest

WORKDIR /app

# Copy root-level source and docs
COPY *.py *.md *.txt /app/

# Copy application packages
COPY agent /app/agent
COPY tts /app/tts
COPY web /app/web

EXPOSE 8000

ENTRYPOINT ["uvicorn", "web.server:app", "--host", "0.0.0.0", "--port", "8000"]
