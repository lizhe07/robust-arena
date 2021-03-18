# robust-arena
Various Robust Tests for Deep Networks

## Dependencies
* pytorch
* foolbox
* jarvis

## Usage
* **Step 0**:

    Prepare a directory for storing models and results, and save your models with `torch.save` as `.pt` files. Each `.pt` file can be loaded by `saved = torch.load(file_pth)`, with `saved` as a dictionary containing keys `'model'` and `'task'`. `saved['model']` should be a PyTorch module that takes images as inputs and returns logits as outputs, `logits = saved['model'](images)`. `images` is a tensor of shape `(N, C, H, W)` with values in $[0, 1]$, `logits` is a tensor of shape `(N, class_num)`. `saved['task']` is a string of dataset name, can be `'MNIST'`, `'CIFAR10'` or `'CIFAR100'` for now.

### Adversarial attacks
* **Step 1**:

    Attack all exported models stored in `[store_dir]/models/exported` on the testing set by `python -m roarena.attack --store_dir [store_dir] --datasets_dir [datasets_dir]`. This command line will attack each model on multiple batches of images, and can deployed onto a cluster for parallel processing. By default `[store_dir]` will be `store` in the current directory, and `[datasets_dir]` will be `vision_datasets` in the current directory. Make sure the datasets have already been downloaded in `[datasets_dir]`, for example `[datasets_dir]/cifar-10-batches-py` for CIFAR10 dataset.

    Advanced usage involves specifying a parameter search space by a file, and call `python -m roarena.attack --spec_pth [spec_pth]`.

* **Step 2**:
    
    Copy the demo notebook `adversarial.robustness.ipynb` to the working directory, and use it to plot figures.