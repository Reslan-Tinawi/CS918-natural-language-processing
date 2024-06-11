# Project stutructre

The project consists of:

1. Single standalone jupyter notebook (`solution.ipynb`) file which include **all** the experiments, model development, data exploration, etc.

2. Data folder: this folder should store the 5 text files of the SemEval data as well as the GloVe embedding file.

3. Utility scripts: to guarnte the notebook is manageable and short, redaundant code such as model definition, text pre-processing logic, metrics plotting and so on, have been moved to separate scripts and then imported in the notebook.

4. Model weights folder: this folder will contain the weight of the models trained. While running the notebook, the best model weights for each of LSTM, LSTM with Attention, and BERT will be stored in this folder. Later, these weights will be used to re-create the model and generate predictions.

5. Sbatch script: this script is used to start a jupyter server on the DCS batch machines.

6. twitter mask: this is a logo of twitter which will be used for generate wordclouds.

# Running instructions

Running this notebook requires availability of GPU for training LSTM models and fine-tuning BERT models.

The notebook has been developed and tested on the [DCS batch compute machines](https://warwick.ac.uk/fac/sci/dcs/intranet/user_guide/batch_compute/).

The `jupyter.sbatch` script file is reposible for starting a jupyter server on the `falcon` machine.

The easist way to connect to DCS machines and run a jupyter notebook is using [VS Code](https://code.visualstudio.com/) along with [remote SSH extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh) and [Jupyter extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)

To run the code, first copy the content of the project to DCS machines. Download the SemEval data and GloVe embedding and place them in the `data` folder. Then ssh into kudu.

First, we connect to the `kudu` machine using VS Code. Pres `Ctrl (⌘) + Shift + P` in vscode and select `Connect to remote SSH`:

![alt text](image.png)

Then, input the following for kudu host machine:

```
kudu.dcs.warwick.ac.uk
```

This will prompt you to input your DCS username and password.

After connecting to `kudu`, change directory to the project directory:

```
cd PROJECT_DIRECTORY
```

Once in the project directory, submit the job for running jupyter notebook as follows:

```
sbatch jupyter.sbatch
```

**NOTE**: please note that while working on the assignment, the `falcon` partition was available. If you try to submit the job but with no success, please consider changing the partition to one of: `gecko` or `eagle`. This might affect the running time of the notebook.

To change partition, in the `jupyter.sbatch` file, modify the 4-th line:

```
#SBATCH --partition=PARTITION_NAME      # Partition you wish to use (see above for list)
```

After successfully running the jobs, two files will be created in the project directory:

1. `jupyter.err`
2. `jupyter.log`

Open the `jupyter.err` file and copy the server URL. An example URL would be something like this:

```
http://falcon-03:11888/tree?token=d53197876aeabda2885256ab38988d5dacc745cf735de6f2
```

Now, open the `solution.ipynb` notebook file and choose `select kernel`:

![alt text](image-1.png)

Paste the jupyter server URL in the input box and you should be connected to a GPU-enabled machine.
