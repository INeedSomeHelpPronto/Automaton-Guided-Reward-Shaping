### To run the experiments:

- Use a conda python 3.7 environment
-`conda create -n agrs python=3.7 && conda activate agrs`
- `pip install -r requirements.txt`
- `python autograph/play/maze_nn_aut.py --workers N --device cpu/cuda:N --checkpoint path/to/checkpoint --checkpoint-every N
--log path/to/tensorboard/log/folder 
path/to/config/file.json`
- Performance may start to degrade if there are too many worker threads. Recommended 4-6.
- Make sure that the root folder (containing autograph and schemas) is the working directory or is on PYTHONPATH
- If you exclude checkpoint-every, it defaults to 1 (checkpoint every episode)
  - Or include `--do-not-save-checkpoint`
- Also, make sure that the folder the checkpoint gets saved to is already created
- For transfer learning, the target for the transfer loads from the "source" checkpoint. In the configuration json for the transfer learning, 
point the `transplant/from` key to the checkpoint you want to load from
- When editing the configuration files, make sure that your editor treats json files as [JSON5](https://json5.org/)
(instructions for [Pycharm](https://www.jetbrains.com/help/idea/json.html#ws_json_choose_version)) 
- The JSON config schema is in `schemas/config_schema.json`
- It can automatically `--stop-after` a certain number of environment steps
- The `--run-name` parameter will substitute a `%s` in the log, checkpoint, and config arguments:
- If you wish to run multiple consecutive runs, call the script `autograph/play/maze_nn_aut_many_runs.py` and include `--stop-after` and `--num-runs` arguments.


  - For example, to run 100 consecutive training runs of 30k steps each, we run the command: `autograph/play/maze_nn_aut_many_runs.py --run-name exp_pit/abcpit_100
--device
cuda:0
--workers
6
--checkpoint-every
2
--num-runs 
3
--stop-after 
30000
--log
runs/mcts/%s
--checkpoint
checkpoints/%s
config/%s.json`
- To run multiple copies of a single experiment on multiple machines, make sure that you use --run-name and %s
  - Create a configuration that conforms to schemas/multiconfig.json
  - Run `autograph/play/run_multi.py --num-runs n --username ssh_username path/to/your/config.json`
- View results in run/total_reward in tensorboard
  - Recommended smoothing of 0.99
- Worker thread 0 will also print out its environment state at the end of every episode.
- If you wish to see every play step, set render to True in autograph/lib/running.py run_episode_generic() function.

- In order to extract log csv's from the 100 training runs, use the `getLogs.py` script. You can specify the base directory, log directory, and out directory for the csvs.

- Use http://merge-csv.com/ to merge your csv files into a single, one header CSV. There is a way to do this in python, this website is quick fix.

- Blind Craftsman Config Folders: `autograph/play/config/mine_medium_static/` `autograph/play/config/mine_big/`
- Treasure Pit Config Folders: `autograph/play/config/exp_pit/`
- Mine Recycler Config Folders: `autograph/play/config/exp_pqr/`
- Mine Maze Config Folders: `autograph/play/config/mine_maze/`

- Use one of the .ipynb notebooks in the evaluation directory to generate graphs. The only change you would make to these notebooks is the file locations.
  