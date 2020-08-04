FROM zheli18/pytorch:1.6.0-cp38-cuda102-1804

# Github username and password
ARG GITHUB_USERNAME
ARG GITHUB_PASSWORD

RUN pip install -U pip setuptools jupyterlab numba

RUN git clone https://$GITHUB_USERNAME:$GITHUB_PASSWORD@github.com/lizhe07/jarvis.git
RUN git clone https://$GITHUB_USERNAME:$GITHUB_PASSWORD@github.com/lizhe07/sim-reg.git
RUN git clone https://$GITHUB_USERNAME:$GITHUB_PASSWORD@github.com/lizhe07/blur-net.git
RUN git clone https://github.com/bethgelab/foolbox.git

RUN pip install -e jarvis sim-reg blur-net foolbox

RUN git clone https://$GITHUB_USERNAME:$GITHUB_PASSWORD@github.com/lizhe07/robust-arena.git
WORKDIR robust-arena
