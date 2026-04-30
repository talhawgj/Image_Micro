# Use the official AWS Lambda Python 3.12 base image
FROM public.ecr.aws/lambda/python:3.12

# 1. Install strictly AL2023-compatible OS dependencies for Headless Chrome
RUN dnf update -y && \
    dnf install -y \
    alsa-lib \
    atk \
    cups-libs \
    gtk3 \
    libXcomposite \
    libXcursor \
    libXdamage \
    libXext \
    libXi \
    libXrandr \
    libXScrnSaver \
    libXtst \
    pango \
    at-spi2-atk \
    libXt \
    nss \
    mesa-libgbm \
    unzip \
    tar \
    gzip && \
    dnf clean all

ENV CHROME_VERSION="131.0.6778.204"
RUN curl -Lo /tmp/chrome.zip "https://storage.googleapis.com/chrome-for-testing-public/${CHROME_VERSION}/linux64/chrome-linux64.zip" && \
    curl -Lo /tmp/chromedriver.zip "https://storage.googleapis.com/chrome-for-testing-public/${CHROME_VERSION}/linux64/chromedriver-linux64.zip" && \
    unzip /tmp/chrome.zip -d /opt/ && \
    unzip /tmp/chromedriver.zip -d /opt/ && \
    rm /tmp/chrome.zip /tmp/chromedriver.zip && \
    ln -s /opt/chrome-linux64/chrome /usr/bin/chrome && \
    ln -s /opt/chromedriver-linux64/chromedriver /usr/bin/chromedriver

# 3. Copy requirements and install Python dependencies
COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the microservice code into the Lambda Task Root
COPY main.py ${LAMBDA_TASK_ROOT}/
COPY config.py ${LAMBDA_TASK_ROOT}/
COPY routes/ ${LAMBDA_TASK_ROOT}/routes/
COPY services/ ${LAMBDA_TASK_ROOT}/services/
COPY utils/ ${LAMBDA_TASK_ROOT}/utils/
COPY schemas/ ${LAMBDA_TASK_ROOT}/schemas/

# 5. Ensure all files are readable by the Lambda execution user
RUN chmod -R 755 ${LAMBDA_TASK_ROOT}

# 6. Tell the Lambda runtime where the Mangum handler is located
CMD ["main.handler"]