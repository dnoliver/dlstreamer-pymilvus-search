FROM docker.io/intel/dlstreamer:2025.0.1.2-ubuntu24

USER root

RUN apt-get update && apt-get install --yes gstreamer1.0-plugins-ugly jq

USER dlstreamer

ENV GST_PLUGIN_PATH=$GST_PLUGIN_PATH/usr/lib/x86_64-linux-gnu/gstreamer-1.0/

WORKDIR /home/dlstreamer/app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

CMD tail -f /dev/null