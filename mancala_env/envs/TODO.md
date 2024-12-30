# Todo next

* Get minimal dependencies for deploying inference
    - Leave out torch gpu stuff
* add some unit tests to test:
    - normal sequence play
    - turn repeat
    - captures
    - game ends correctly
    - game ends after turn that would otherwise repeat
    - don't put gems in opponent mancala
    - starting state is correct
    - Note on implementation: will need a way to set the opponent policy to be deterministic and known
* check that self-play against the previously trained model will work.
    - i.e. ensure you can set the opponent policy from a previously trained model
