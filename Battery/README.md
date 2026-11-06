This file contains the Thevenin model for the Starling 2 Drone we are simulating on. The components found on board the Starling are treated as loaded resistors in parallel, each with their respective power consumption ratings. The discharge of the battery is modeled using a 4th degree polynomial:  v = (4.15 - 0.4 * (1 - soc) - 0.3 * (1 - soc) ** 2 + 0.5 * (1 - soc) ** 3 - 0.8 * (1 - soc) ** 4 + 0.15 * soc). SOC stands for the state of charge (The battery fully charged or depleted) for a single cell.

The thevenin models are calculated to find the load currents and voltage drops across the system. The script reads measurements from the drone when undergoing flight using model predictive control and returns a .csv file containing the time, thevenin voltage, voltage sag (esc), load current, motor power, payload power, and state of charge).

The .csv file is then plotted using the plotter script to display the current draw of the drone over time while flying across the course.


