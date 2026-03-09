# Create directory structure

```bash
mkdir -p coco/images && cd coco

# Download annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip

# Download images (choose one or all)
wget http://images.cocodataset.org/zips/train2017.zip   # ~18GB
wget http://images.cocodataset.org/zips/val2017.zip     # ~1GB  ← start here
wget http://images.cocodataset.org/zips/test2017.zip    # ~6GB

unzip val2017.zip -d images/
unzip train2017.zip -d images/
```