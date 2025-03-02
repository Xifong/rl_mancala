# Todo next

## Training Improvements
~* Try out RLZoo for hyperparameter tuning~
* command line options to train (i.e. set opponent and starting model etc)
* Focus on correctness and testing of env/training
    - Latest bugs have shown me that having a more robust way to check that the things are genuinely correct is critical
    - Rendering + unit tests:
    - normal sequence play
    - turn repeat
    - captures
    - game ends correctly
    - game ends after turn that would otherwise repeat
    - don't put gems in opponent mancala
    - starting state is correct
    - Note on implementation: will need a way to set the opponent policy to be deterministic and known
    - Possible refactor of environment to reduce complexity and chance of bugs
* When saving model through EvalCallback, save the replay buffer too for less disjointed initial training.
* More reward tweaking (should there be a reward for captures/getting to play again?)

## Service Improvements
~~* Switch all the POST to PUT requests to reflect that these are idempotent~~
~~* and getInitialState can probably be a GET request really~~
~~* Seems like there's a backend bug where it plays on the wrong side :/~~
~~* Backend for next_state should check the body conforms to expectations~~
~~* Backend for get requests should also use pydantic validation~~
~* Backend for next_move should check the body conforms to expectations~
~~* Backend should use strictest validation mode~~
~* Backend should say if game is over~
~* Backend should say what move indexes are valid~
~* infer_from_observation should rotate board perspective as needed~
~* New endpoint for combined (play next move and get next)~
* Backend should reject playing moves from the wrong player/ensure the move is played on the correct player
* Improve loading of models into inference api:
    * Instead of loading the prod saved model into the container directly, provision a bucket and have a build script to push the latest model there. Have the running service use the latest model from the bucket
    * Just export the policy, don't even export the model
    * Maybe above will allow us to eliminate some more inference dependencies?
* have all outs go to `out` dir
* Split api out of mancala_agent_pkg


## Code Quality
~* mancala env library should not configure logging handlers opaquely~
* load logging configs in from config files
* All the inline TODOS!
