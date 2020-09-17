FROM zheli18/pytorch:1.6.0-cp38-cuda102-1804

# Github username and password
ARG GITHUB_USERNAME
ARG GITHUB_PASSWORD

RUN pip install -U pip setuptools jupyterlab numba

RUN git clone https://$GITHUB_USERNAME:$GITHUB_PASSWORD@github.com/lizhe07/jarvis.git
RUN git clone https://$GITHUB_USERNAME:$GITHUB_PASSWORD@github.com/lizhe07/sim-reg.git
RUN git clone https://$GITHUB_USERNAME:$GITHUB_PASSWORD@github.com/lizhe07/blur-net.git
RUN git clone https://$GITHUB_USERNAME:$GITHUB_PASSWORD@github.com/lizhe07/augmix.git
RUN git clone https://github.com/bethgelab/foolbox.git

RUN pip install -e jarvis
RUN pip install -e sim-reg
RUN pip install -e blur-net
RUN pip install -e augmix
RUN pip install -e foolbox

RUN mv /robust-arena/foolbox_fix/brendel_bethge.py /foolbox/foolbox/attacks

RUN git clone https://$GITHUB_USERNAME:$GITHUB_PASSWORD@github.com/lizhe07/robust-arena.git
WORKDIR robust-arena
