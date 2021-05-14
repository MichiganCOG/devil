# DEVIL Dataset Construction

This document explains how to re-generate the video and mask splits used in the DEVIL benchmark. It is mainly for
completeness, since the pre-generated splits are already available online and can be downloaded.

## Instructions

### Setup

```
# Download weights for models used by DEVIL
./scripts/download-weights.sh
```

### Download Flickr videos and construct clips

```
python -m src.main.download_flickr_videos
python -m src.main.create_flickr_clips
```

### Score/classify Flickr clips

```
python -m src.main.score_flickr_clips
```

### Render occlusion masks

```
splits=( \
  devil_fgs-l_fgm-l_fgd-l \
  devil_fgs-l_fgm-l_fgd-h \
  devil_fgs-l_fgm-h_fgd-l \
  devil_fgs-l_fgm-h_fgd-h \
  devil_fgs-h_fgm-l_fgd-l \
  devil_fgs-h_fgm-l_fgd-h \
  devil_fgs-h_fgm-h_fgd-l \
  devil_fgs-h_fgm-h_fgd-h \
)
for split in ${splits[@]}; do python -m src.main.render_fvi_masks $split; done
```

### Construct DEVIL splits

```
python -m src.main.create_devil_video_lists
for file in $(ls video-lists/fvi-masks); do python -m src.main.create_devil_split masks ${file%.txt}; done

# Create source video splits. For some reason, only the "all" split got shuffled for our publication...
python -m src.main.create_devil_split frames all
python -m src.main.create_devil_split frames bsm-h --no_shuffle
python -m src.main.create_devil_split frames bsm-l --no_shuffle
python -m src.main.create_devil_split frames cm-h --no_shuffle
python -m src.main.create_devil_split frames cm-l --no_shuffle
```

### OPTIONAL: Archive DEVIL splits in tar files

You can save the DEVIL splits in `.tar` archives, which are a bit more NFS-friendly than the normal directory structure.
To create them, run the following script:

```
./scripts/archive-devil-splits.sh
```
