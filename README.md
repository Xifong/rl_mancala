# Cheating at Mancala

Over Christmas 2024, I've been playing Mancala with my family. This is a project I've put together to feed me moves to beat them with. It's a pretty simple game, so the custom gymnasium environment under `./mancala_env/` was quick to write and taking stock DQN without any tuning from stable_baselines3 has been working out so far. What I call the greedy agent achieves a 99+% win rate against random moves, and training against the greedy agent so far I've gotten a similarly high win rate in evaluation. 

## First Time Install
Use at risk (setup script untested). Also assumes you can use CUDA.
```bash
./build/first_time/local.sh
```

## Train
```bash
export TARGET_VENV=venv && ./build/env/local_build.sh && python3 -m pkg.mancala_agent_pkg.model.train
```
The agent will be evaluated periodically during training, with the best on-policy evaluation (by mean reward) being saved into `./saved_models/`. View a plot of training statistics under `./last_run/plots.png`.

### Save model and regenerate plots

```bash
python3 -m pkg.mancala_agent_pkg.model.save
```

This is already done at the end of training, but can be forced to iterate on the plots themselves.

## Run inference API server
From any venv (doesn't matter):
### Locally
```bash
export TARGET_VENV="inference_venv" && ./build/env/local_build.sh && python -m pkg.mancala_agent_pkg.inference_api.server 
```

### Locally in Docker
```bash
./build/api/build.sh && PORT=8000 && docker run -d -p 8000:${PORT} -e PORT=${PORT} mancala 
```
Then to debug there are some options:
```bash
docker ps # to get process number
docker logs -f <process number>
docker run -it --entrypoint=/bin/bash mancala
# Can obviously then send requests to localhost:8000
```

## Mancala custom env
From development venv:
```bash
pytest && python3 ./mancala_env/envs/mancala.py
```

Unit tests are (very much) incomplete atm.

### Build and install mancala env
```bash
export TARGET_VENV=inference_venv && ./build/env/local_build.sh
```
