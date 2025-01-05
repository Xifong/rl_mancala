The listed requirements are not exhaustive. I've taken pains to minimise the size of the dependencies required on the production container, so you have to install the cpu-only version of pytorch from a non-standard location:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

After this, the rest of the requirements are in `inference_requirements.txt`. The scripts under the `build` directory should automate all of this though, so most likely you won't need to be doing it.
