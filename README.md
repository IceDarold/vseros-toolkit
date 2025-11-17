# vseros-toolkit

### Image features (embeddings + fast stats)
```python
from common.features.img_index import build_from_dir
from common.features.img_embed import build as img_embed
from common.features.img_stats import build as img_stats
from common.features import store, assemble

id2imgs = build_from_dir("data/images", ids=[...], pattern="{id}/*.jpg", max_per_id=4)
FS = store.FeatureStore()
FS.add(img_stats(train, test, id_col="id", id_to_images=id2imgs))             # быстрый fallback
FS.add(img_embed(train, test, id_col="id", id_to_images=id2imgs,              # CNN/ViT
                 backbone="resnet50", image_size=224, agg="mean", pool="avg",
                 device="auto", precision="auto", dtype="float16"))
X_dense_tr, X_dense_te, catalog = assemble.make_dense(FS, include=FS.list())
```
