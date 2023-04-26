set -e

# Install Miniconda

wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh
bash Miniconda3-py39_4.12.0-Linux-x86_64.sh -b -p
rm -rf Miniconda3-py39_4.12.0-Linux-x86_64.sh
PATH=$HOME/miniconda3/bin:$PATH
conda init

# Clone and install the repo

# Check for ssh key file
if [ ! -f ~/.ssh/id_rsa ]; then
    echo "~/.ssh/id_rsa does not exist on the remote instance. Please ensure key is provided and re-run script."
    exit 1
fi

# Set correct permissions for the ssh key and add it to the ssh-agent
chmod 600 ~/.ssh/id_rsa
eval `ssh-agent -s`
ssh-add ~/.ssh/id_rsa

# Skip over prompt to add github to known_hosts
ssh-keyscan -H github.com >> ~/.ssh/known_hosts

# Clone, create conda env, and install dependencies
git clone git@github.com:afterless/cs4644_final.git
ENV_PATH=~/cs4644_final/.env/
cd $ENV_PATH
conda create -p $ENV_PATH python=3.10 -y
conda install -p $ENV_PATH pytorch=1.12.0 torchtext torchdata torchvision -c pytorch -y
conda run -p $ENV_PATH pip install -r requirements.txt

# Add the ssh key to the ssh-agent each time
echo '
# Add github service account SSH key to agent
if [ -z "$SSH_AUTH_SOCK" ] ; then
  eval `ssh-agent -s`
  ssh-add ~/.ssh/cs4644_final_ssh
fi
' >> ~/.bashrc

# Activate the mlab2 virtualenv each time
echo '
conda activate ~/cs4644_final/.env/
' >> ~/.bashrc