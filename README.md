# Deep Q-Networks
Implementation of Deep Q-Networks on Atari games with *Tensorflow-gpu 2.0* in *python 3.6*

### Requirements

- Python 3.*
- Tensorflow 2.0 
- gym

```bash
conda create -n virEnv python=3.6 numpy scipy
conda activate virEnv
pip install -q tensorflow-gpu==2.0.0-alpha0
pip install -U 'gym[all]'
```

### Summary

1. DQN: Q-Learning but with a deep neural network as a function approximator.
2. 


### Log
1.  Create the neural network model which will be estimator of Q function and target function. Be cautious of the input and output shape.
2.  Note that, the input state of the model is preprocessed, and we may need to print out or output the input state to visualize and validate it.
3.  

### Reference

