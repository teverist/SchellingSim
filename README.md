# COMP6216 Simulation Modelling for Computer Science Group Project


# Installation

### Linux

1. Clone the repository
```bash
git clone https://github.com/teverist/SchellingSim.git
```

2. Create a virtual environment using the following command:
```bash
python -m venv venv
```

3. Activate the virtual environment using the following command:
```bash
source venv/bin/activate
```

4. Install the required packages using the following command:
```bash
pip install -r requirements.txt
```

### Windows

1. Clone the repository
```bash
git clone https://github.com/teverist/SchellingSim.git
```

2. Create a virtual environment using the following command:
```bash
python -m venv venv
```

3. Activate the virtual environment using the following command:
```bash
venv\Scripts\activate
```

4. Install the required packages using the following command:
```bash
pip install -r requirements.txt
```

# How it works

Suppose there are two types of agents: A and B. These agents represent a member of different groups. Two populations of these agents are placed randomly on a grid. Each cell on the grid is either occupied by an agent or empty.

An agent satisfied of their location is surrounded by at least a fraction $t$ of agents of the same type. When an agent is not satisfied, it can moved to any empty cell in the grid.