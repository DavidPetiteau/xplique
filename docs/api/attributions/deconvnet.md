# Deconvnet

!!!abstract "Tutorial"
    <p style="text-align: center;">[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1qBxwsMILPvQs3WLLcX_hRb3kzTSI4rkz?authuser=1)</p>

## Example

```python
from xplique.attributions import DeconvNet

# load images, labels and model
# ...

method = DeconvNet(model)
explanations = method.explain(images, labels)
```

{{xplique.attributions.deconvnet.DeconvNet}}
