# Autoencoders for Unsupervised Anomaly Segmentation in Brain MR Images: Dockerized launch

## Usage

```bash
docker build . -t uadbm:local

docker run --rm -v <abspath-to-home>/content/data/Brainweb/lesions/severe:/input -v <abspath-to-home>/man:/output uadbm:local --nii /input/t2_ai_msles2_1mm_pn5_rf40.nii.gz --outdir /output
```

## Test dataset

run this script in container:
```bash
from utils.brainweb_download import download_brainweb_dataset
from pathlib import Path

download_brainweb_dataset(
    base_dir=Path('./content/data/Brainweb'),
    name="",
    institution="",
    email=""
)
```
