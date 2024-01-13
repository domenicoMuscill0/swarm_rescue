## How to run the program?

`python3 -m venv swarm_rescue // Create the vatual environment`

`source swarm_rescue/bin/activate`

`pip install -r requirements.txt`


The communication part is in the : `file /swarm_rescue/src/swarm_rescue/solutions/drone_communication.py`

Change the import part in launcher.py to `from solutions.drone_communication import MyDroneEval` to load the communication version