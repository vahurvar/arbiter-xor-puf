FROM ubuntu
RUN apt update && apt upgrade -y && apt install python3-pip -y
WORKDIR /puf
COPY requirements.txt .
RUN pip3 install -r requirements.txt
COPY . .
ENTRYPOINT ["sh"]
CMD ["run-all.sh"]