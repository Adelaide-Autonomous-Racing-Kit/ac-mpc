# Assetto Corsa Model Predictive Control
Full stack autonomous racing solution based on deeplearnt perception and MPC using ACI.

## Installation
Make sure ACI is installed and working correctly by following the steps outlined [here](https://github.com/Adelaide-Autonomous-Racing-Kit/ac-interface?tab=readme-ov-file#installation)

To use the diagnostics dashboard install the PyQt6 system dependency: `sudo apt-get install libxcb-cursor0`

To download assets required to run our baseline examples run `bash scripts/download_assets.sh <proton/crossover>`

## Run
### Bare Metal
To run an agent on directly on the host machine execute the command `python src/main.py --config configs/<your_agent_config.yaml>`

### Docker (Experimental)
We also have beta support for running agents inside of docker containers using a helper script.
To do this execute `bash scripts/run.sh configs/<your_agent_config.yaml>`.

**NOTE:** In our experience this does make execution slightly less performant and there are some teething issues around how processes in the container manage processes on the host machine.


## Usage
For more detailed documentation refer to our [website](https://adelaideautonomous.racing/docs-acmpc/)

## Citation
If you use AC-MPC in your research or quote our baselines please cite us in your work:
```BibTeX
@misc{bockman2024aarkopentoolkitautonomous,
      title={AARK: An Open Toolkit for Autonomous Racing Research}, 
      author={James Bockman and Matthew Howe and Adrian Orenstein and Feras Dayoub},
      year={2024},
      eprint={2410.00358},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2410.00358}, 
}
```
