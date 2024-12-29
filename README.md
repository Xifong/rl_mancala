Train:
```bash
python3 model/train.py && wslview ./last_run/plots.png
```

Regenerate plots:
```bash
python3 model/save.py && wslview ./last_run/plots.png
```

Infer:
```bash
python3 model/infer.py
```

Test env:
```bash
pytest && python3 ./mancala_env/envs/mancala.py
```
