FROM dschng/sdp-tph:v5.0
WORKDIR /usr/src/app
RUN pip install laspy
ENTRYPOINT []
CMD []
USER 0
WORKDIR /
# ENTRYPOINT ["python","./main.py"]
# CMD ["file_dir" "input_file_name" "file_type"]

# docker build -t test-new-image .