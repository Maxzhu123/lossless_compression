touch ~/.no_auto_tmux
echo 'set -g mouse on' >> ~/.tmux.conf

apt update -y
apt install -y build-essential

wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
source .bashrc
conda config --set auto_activate_base false
source .bashrc

mamba create -y -n bitsparse python=3.14.*
mamba activate bitsparse

git clone https://github.com/Maxzhu123/lossless_compression.git
cd lossless_compression
pip install -r requirements.txt
cd baselines
./setup.sh
cd ..