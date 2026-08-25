FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    vim \
    lsb-release \
    wget \
    gnupg \
    software-properties-common \
    python3 \
    python3-venv \
    python3-pip \
    python3-dev \
    automake \
    cmake \
    flex \
    bison \
    tmux \
    docker.io \
    doxygen \
    xsltproc \
    docbook-xsl \
    python3-sphinx \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

RUN curl -sSL -o /tmp/llvm.sh https://apt.llvm.org/llvm.sh && \
    chmod +x /tmp/llvm.sh && \
    /tmp/llvm.sh 19 all && \
    rm /tmp/llvm.sh && \
    update-alternatives --install /usr/bin/clang clang /usr/bin/clang-19 100 && \
    update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-19 100 && \
    update-alternatives --install /usr/bin/clangd clangd /usr/bin/clangd-19 100 && \
    update-alternatives --install /usr/bin/llvm-profdata llvm-profdata /usr/bin/llvm-profdata-19 100 && \
    update-alternatives --install /usr/bin/llvm-cov llvm-cov /usr/bin/llvm-cov-19 100 && \
    update-alternatives --install /usr/bin/llvm-nm llvm-nm /usr/bin/llvm-nm-19 100

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc-$(gcc --version|head -n1|sed 's/\..*//'|sed 's/.* //')-plugin-dev \
    libstdc++-$(gcc --version|head -n1|sed 's/\..*//'|sed 's/.* //')-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/AFLplusplus/AFLplusplus && \
    cd AFLplusplus && \
    make source-only && \
    make install && \
    cd .. && \
    rm -rf AFLplusplus

RUN apt-get update && \
    apt-get install -y locales && \
    locale-gen en_US.UTF-8 && \
    update-locale LANG=en_US.UTF-8 && \
    rm -rf /var/lib/apt/lists/*
ENV LANG=en_US.UTF-8
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /afk

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

RUN curl -qsL 'https://install.pwndbg.re' | sh -s -- -t pwndbg-gdb

CMD ["/bin/bash"] 