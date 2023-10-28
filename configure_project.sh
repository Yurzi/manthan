#!/bin/sh

# update git submodule
git submodule update --init --recursive

# configure deps
chmod +x ./configure_dependencies.sh
./configure_dependencies.sh

# install pip requirements
pip install -r requirements.txt

# download oss-cad-suite
if [ ! -d "oss-cad-suite" ]; then
    wget -c https://github.com/YosysHQ/oss-cad-suite-build/releases/download/2023-10-28/oss-cad-suite-linux-x64-20231028.tgz
    tar -xzf oss-cad-suite-linux-x64-20231028.tgz
    rm oss-cad-suite-linux-x64-20231028.tgz
fi