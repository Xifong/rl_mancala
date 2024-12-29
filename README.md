Train: 
```bash
python3 model/train.py && wslview logs/plots.png
```

Regenerate plots:
```bash
python3 model/view_logs.py && wslview logs/plots.png
```

Infer:
```bash
python3 model/infer.py
```

Test env:
```bash
pytest && python3 ./mancala_env/envs/mancala.py
```
