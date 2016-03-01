## Data preparation

### Selecting training images

The ``sed`` tool can be used to select every *n*-th input image for masking. For
example, to select every 10th image from the bell dataset:

```bash
for fn in $(ls data/bell/input/*.JPG | sed -n '0~10p'); do
  echo $fn
  convert $fn -colorspace gray $(dirname $fn)/../trimap/$(basename $fn .JPG).png
done
```


