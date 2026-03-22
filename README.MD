# Value Iteration for a Finite MDP

This repository is a small coursework project about value iteration for a finite discounted Markov Decision Process. The aim of the project is to write a reusable Python package, test the main functions, and show one working example with the grid world problem from the assignment.

The code is kept simple and split into a few files. The file `mdp.py` defines the MDP and transition structure. The file `policy.py` contains the action-value calculation and greedy policy extraction. The file `solver.py` contains the main value iteration algorithm. There is also a notebook called `Report_Grid_World.ipynb` which gives a short demonstration of the grid world example.

To use the package locally, go to the project folder and install it with:

```bash
pip install -e .
```

If you also want to run the tests and the notebook, install the development requirements with:

```bash
pip install -r requirements-dev.txt
```

After installation, the tests can be run with:

```bash
pytest -v
```

The notebook can be opened in Jupyter Notebook and used to check the grid world values, policy, and convergence result. It is better to install the package first before running the notebook, because the notebook imports `value_iteration` as an installed package.

This project is not intended to be a large software package. It is mainly a clear implementation for study, testing, and coursework discussion.

License: MIT
