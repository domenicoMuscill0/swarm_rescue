## How to run the program?

`python3 -m venv swarm_rescue // Create the vatual environment`

`source swarm_rescue/bin/activate`

`pip install -r requirements.txt`


The communication part is in the : `file /swarm_rescue/src/swarm_rescue/solutions/drone_communication.py`

Change the import part in launcher.py to `from solutions.drone_communication import MyDroneEval` to load the communication version

Some idea on how to design the communication part of the codes:

I was thinking of creating a json-like structure of the message to be comunicated that will gonna be completed during the control function depending on what happens. For example i have solved the problems to recognize when a drone enters a no-gps zone and make him move using the odometer sensor instead of the gps. I have also found a way to understand when a drone enters a destruction zone: in these cases we can manage to send specific flags (or more simply put a new key in the message dictionary) that can make the others have more insights on where they are

With a practical example: a drone enters a no-gps zone, so he will compute its self.gps_val attribute using the odometer sensor instead of the standard gps. If we also use the communication system a drone in this particular zone can set its position as the average position among the one he computed with the odometer and the ones derived from computing the position given the other drones' positions using standard linear algebra

This will make the estimate on the position benefit from the presence of other drones and be more accurate

The thing i am more curious about is the effect of sharing informations about the grid/graph. I see that most of the times the drones spend a lot of time searching places in which others passed by some time before, so we could achieve better rescue speed by doing so

 Also we have a problem when a drone is carrying a body but other drones step on its path, blocking him from delivering the body. I think it would be optimal if we share with the communication system the coordinates of the self.following attribute of the "ambulance" drone so that all the ones that will receive it will try to stay as far as possible from that segment
