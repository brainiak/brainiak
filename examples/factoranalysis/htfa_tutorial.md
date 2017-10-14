This tutorial uses the *MIND Tools* docker container; installation instructions [here](https://github.com/Summer-MIND/mind-tools).

After installing the container, activate it using:
```
docker start MIND && docker attach MIND
```
To close the container, type `ctrl + d` in terminal.

Then open the Jupyter notebook using
```
jupyter notebook --port=9999 --no-browser --ip=0.0.0.0 --allow-root
```

Navigate to `localhost:9999` in your browser and open the tutorial notebook.
