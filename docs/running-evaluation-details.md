# Running Evaluation - Details

This document describes how to perform evaluation on our DEVIL splits. It covers various details such as using `.tar`
files, environment variables, and Slurm.

## Computing Evaluation Features

Note: Evaluation features take up a lot of space (~50GB per DEVIL split x 5 DEVIL splits). Before running the following
commands, make sure that `<project root>/eval-data` points to a disk with sufficient capacity (you can use a symbolic
link if necessary).

First, make sure that the desired DEVIL splits corresponding to real videos exist as normal directories in
`datasets/devil`. If not, download and extract them:

```bash
for split in ( all bsm-h bsm-l cm-h cm-l  ); do
  python -m src.main.download_devil_splits flickr-$split -e;
done
```

Then, generate the evaluation features:

```bash
for split in ( all bsm-h bsm-l cm-h cm-l  ); do
  source scripts/compute-devil-evaluation-features.sh flickr-$split;
done
```

### OPTIONAL: Archive evaluation features

The evaluation features can be stored in `.tar` files, which are slightly more NFS-friendly than normal directories. To
create them, run the following script:

```bash
./scripts/archive-evaluation-features.sh
```

## Downloading Inpainting Results

Previously generated inpainting results can downloaded with `download_inpainting_results.py`. For instance, the
following example downloads CPNet predictions on the `fvi-fgd-h` split:

```bash
python -m src.main.download_inpainting_results cpnet flickr-all fvi-fgd-h
```

For the list of supported models and DEVIL splits, refer to `src/main/download_inpainting_results.py`.  Results will be
saved under the `inpainting-results/devil` folder, and will overwrite any existing results.

## Running Evaluation

**NOTE:** The commands in this section should be run from the `slurm` directory.

The `submit-evaluate-inpainting.py` script runs evaluation metrics on our data. Below is an example of running this 
script on CPNet predictions:

```bash
python submit-evaluate-inpainting.py cpnet flickr-all fvi-fgd-h -m local
```

More generally, given a call of the following form:

```bash
python submit-evaluate-inpainting.py <model> <source split> <mask split> -m local
```

the script will look for results in `<project root>/inpainting-results/devil/<source split>_<mask split>/<model>` and 
compare them to the ground-truth data at `<project root>/datasets/devil/<source split>` and 
`<project root>/datasets/devil/<mask split>`.

If you have predictions for your own model (e.g., at 
`<project root>/inpainting-results/devil/flickr-all_fvi-fgd-h/mymodel`), your can use our evaluation script on them.

### FID

For some reason, the evaluation script usually fails when FID is computed alongside the other metrics; for this reason,
FID is excluded from the default configuration of `submit-evaluate-inpainting.py` and should be computed in a separate
run. The following example computes FID and adds it to the existing CPNet evaluation results:

```bash
python submit-evaluate-inpainting.py cpnet flickr-all fvi-fgd-h -m local -a ::include fid ::update
```

### OPTIONAL: Advanced features

If you want to use the `.tar` files that were optionally generated (i.e., for the DEVIL datasets and evaluation 
features), use the `--use_tar` flag:

```bash
python submit-evaluate-inpainting.py cpnet flickr-all fvi-fgd-h -m local --use_tar
```

Environment variables can be passed in as string pairs with the `-e` flag. The following example runs evaluation with a
custom scratch storage location (which is used to temporally store data during evaluation):

```bash
python submit-evaluate-inpainting.py cpnet flickr-all fvi-fgd-h -m local -e SCRATCH_ROOT /tmp/$USER
```

This script also supports Slurm job allocation, which can be done by removing the `-m local` flag:

```bash
python submit-evaluate-inpainting.py cpnet flickr-all fvi-fgd-h
```

`sbatch` arguments can be passed in as string pairs with the `-s` flag. This illustrative example runs evaluation on a
custom partition:

```bash
python submit-evaluate-inpainting.py cpnet flickr-all fvi-fgd-h -s partition my-partition
```

If necessary, `-s` and `-e` can be used at the same time.

## Printing Evaluation Results

The following script writes all quantitative results to `inpainting-results-quantitative-summary.tsv` as a tab-separated
text table:

```bash
./scripts/print-quant-results-table.sh
```
