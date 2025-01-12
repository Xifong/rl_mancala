# Todo next

* Try out RLZoo for hyperparameter tuning
* command line options to train (i.e. set opponent and starting model etc)
* Focus on correctness and testing of env/training
    - Latest bugs have shown me that having a more robust way to check that the things are genuinely correct is critical
    - Rendering + unit tests
    - Possible refactor of environment to reduce complexity and chance of bugs
* When saving model through EvalCallback, save the replay buffer too for less disjointed initial training.
* Improve loading of models into inference api:
    - Instead of loading the prod saved model into the container directly, provision a bucket and have a build script to push the latest model there. Have the running service use the latest model from the bucket
    - Just export the policy, don't even export the model
    - Maybe above will allow us to eliminate some more inference dependencies?

* have all outs go to `out` dir
* Split api out of mancala_agent_pkg
* add some unit tests to test:
    - normal sequence play
    - turn repeat
    - captures
    - game ends correctly
    - game ends after turn that would otherwise repeat
    - don't put gems in opponent mancala
    - starting state is correct
    - Note on implementation: will need a way to set the opponent policy to be deterministic and known
* All the inline TODOS!
