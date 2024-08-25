import os
from time import time
from lightning_sdk import Studio, Machine
from lightning_cloud.utils import add_s3_connection

HF_TOKEN = os.environ.get("HF_TOKEN")
assert HF_TOKEN != "PUT_YOUR_OWN_TOKEN"


studio = Studio(name="Downloader", teamspace="Audio-Project")
studio.start(machine=Machine.DATA_PREP)


t0 = time()
studio.run("sudo apt-get install git-lfs")
studio.run("pip install -U 'huggingface_hub[cli]'")
studio.run("git config --global credential.helper store")
studio.run(f"huggingface-cli login --token {HF_TOKEN} --add-to-git-credential")
studio.run(f"git clone --progress https://huggingface.co/datasets/SLPRL-HUJI/HebDB")
print(time() - t0)

studio.stop()
