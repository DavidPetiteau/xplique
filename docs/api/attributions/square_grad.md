# Square Grad

!!!abstract "Tutorial"
    <p style="text-align: center;">[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/14c0tb_MMNQzpCFyTtaCgQUfG1OpnFPI0?authuser=1)</p>

Similar to SmoothGrad, Square Grad average the square of the gradients.

$$
\phi_x = \underset{\xi ~\sim~ \mathcal{N}(0, \sigma^2)}{\mathbb{E}}
            \Big{[}\Big{(}
             \frac { \partial{S_c(x + \xi)} } { \partial{x} }
             \Big{)}^2\Big{]}
$$

with $S_c$ the unormalized class score (layer before softmax). The $\sigma$ in the formula is controlled using the noise
parameter.

## Example

```python
from xplique.attributions import SquareGrad

# load images, labels and model
# ...

method = SquareGrad(model, nb_samples=50, noise=0.15)
explanations = method.explain(images, labels)
```

{{xplique.attributions.SquareGrad}}
