# Guided Backpropagation

!!!abstract "Tutorial"
    <p style="text-align: center;">[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/16cmbKC0b6SVl1HjhOKhLTNak3ytm1Ib1?authuser=1)</p>

## Example

```python
from xplique.attributions import GuidedBackprop

# load images, labels and model
# ...

method = GuidedBackprop(model)
explanations = method.explain(images, labels)
```

{{xplique.attributions.guided_backpropagation.GuidedBackprop}}
