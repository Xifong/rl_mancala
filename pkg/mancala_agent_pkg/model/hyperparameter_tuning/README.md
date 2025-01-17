# Performing hyperparameter tuning

Add the following lines or whatever is currently equivalent from `model/train.py` into your file equivalent to `venv/lib/python3.12/site-packages/rl_zoo3/import_envs.py`:
```python
# Import custom Mancala env
from typing import Any
import mancala_env
import pkg.mancala_agent_pkg.model.opponent_policy as op


OPPONENT_MODEL_NAME = "opponent"


opponent_policy = op.get_saved_opponent_policy(OPPONENT_MODEL_NAME, deterministic=False)

register(
    id="Mancala-v0",
    kwargs={"seed": None, "opponent_policy": opponent_policy, "is_play_mode": False},
    entry_point="mancala_env.envs:MancalaEnv",
)
```
Then run for example:
```bash
python -m rl_zoo3.train --algo dqn --env Mancala-v0 -n 50000 -optimize --n-trials 1000 --conf-file=pkg/mancala_agent_pkg/model/hyperparameter_tuning/dqn_mancala_config.yml --log-folder=./last_rl_zoo3_run
```

## References
Custom envs with rl_zoo3 (2025-01-12): https://rl-baselines3-zoo.readthedocs.io/en/master/guide/custom_env.html
