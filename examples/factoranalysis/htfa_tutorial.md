This tutorial uses the *BrainIAK* docker container; installation instructions [here](https://github.com/brainiak/brainiak).

After installing the container, activate it using:
```
docker start demo && docker attach demo
```
To close the container, type `ctrl + d` in terminal.

Then open the Jupyter notebook using
```
jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root
```

Navigate to `localhost:8888` in your browser and open the tutorial notebook.
