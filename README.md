# Guided Masking for Photogrammetry

**Experimental**

This repo contains some experimental code for a machine-learning approach to
photomasking for photogrammetry.

## Data preparation

### Selecting training images

The ``sed`` tool can be used to select every *n*-th input image for masking. For
example, to select every 10th image from the bell dataset:

```bash
for fn in $(ls data/bell/input/*.JPG | sed -n '0~10p'); do
  echo $fn
  convert $fn $(dirname $fn)/../trimap/$(basename $fn .JPG).png
done
```

