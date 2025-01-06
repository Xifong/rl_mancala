# Cheating at Mancala

Over Christmas 2024, I've been playing Mancala with my family. This is a project I've put together to feed me moves to beat them with. It's a pretty simple game, so the custom gymnasium environment under `./mancala_env/` was quick to write and taking stock DQN without any tuning from stable_baselines3 has been working out so far. What I call the greedy agent achieves a 99+% win rate against random moves, and training against the greedy agent so far I've gotten a similarly high win rate in evaluation. 

Dependency installation (training env):
```bash
export venv_name="training_venv"
export dir="model"
python3 -m venv "$venv_name"
source "./$venv_name/bin/activate"
pip install -r "./$dir/requirements.txt"
pip install -e .
```
and the same for the inference but with a difference venv name:

```bash
export venv_name="inference_venv"
export dir="inference_api"
```

## Train

```bash
python3 model/train.py
```
The agent will be evaluated periodically during training, with the best on-policy evaluation (by mean reward) being saved into `./saved_models/`. View a plot of training statistics under `./last_run/plots.png`.

## Save model and regenerate plots

```bash
python3 model/save.py
```

This is already done at the end of training, but can be forced to iterate on the plots themselves.

## Infer
First change the used model in `./model/infer.py` to be a model you've trained using the above.

```bash
python3 model/infer.py
```

## Run inference API server
### Locally
```bash
./build/env/local_build.sh && python -m pkg.mancala_agent_pkg.inference_api.server 
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

## Test Mancala custom env

```bash
pytest && python3 ./mancala_env/envs/mancala.py
```

Unit tests are (very much) incomplete atm.

### Build and install mancala env
```bash
./build/env/local_build.sh
```
