FROM python:3.7
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
RUN python ./app/create_model.py
EXPOSE 8180
COPY ./docker-entrypoint.sh /
RUN chmod +x /docker-entrypoint.sh
ENTRYPOINT ["/docker-entrypoint.sh"]
