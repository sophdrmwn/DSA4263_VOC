FROM python:3.10
WORKDIR /DSA4263_VOC
COPY . /DSA4263_VOC
RUN pip3 install -r requirements.txt
EXPOSE 5000
CMD ["python3", "app.py"]