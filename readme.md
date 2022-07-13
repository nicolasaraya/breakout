# Breakout
![Imgur Image](https://tcf.admeen.org/game/17500/17474/400x246/atari-breakout.jpg)

## Usage
* **Train DQN:** 
    * `$ python main.py --test_dqn`  with 10000 episodes
    * `$ python main.py --test_dqn N`  with N episodes

* **Test Agent:**
    * `$python main.py --train_dqn` train without render
    * `$ python main.py --train_dqn --render` train with render


## requirements:
- VS Build Toolkit (windows)
- pip install tensorflow
- pip install ale-py
- pip install gym
- pip install numpy
- pip install baselines
- pip install opencv-python
- pip install pandas
- pip install matplot
- pip install tqdm
- pip install keras
- pip instal IPython
- `ale-import-roms roms`