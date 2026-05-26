# I. Gaussian Distribution and its gradient

## Multivariate Gaussian Density Function

$$
\mathcal{N} (x \vert \mu, \Sigma) = \frac{\exp( - \frac{1}{2} (x - \mu)^T \Sigma ^{-1} (x - \mu) )}{\sqrt{(2\pi)^k \vert \Sigma \vert }}
$$

Where: 

$$
\begin{cases}
x: \text{data}, \\
\mu: \text{mean matrix}, \\
\Sigma: \text{covariance matrix}, \\
k:  \text{dimensions of data.}
\end{cases}
$$

## Multivariate Gaussian Log Density Function

$$
\log(\mathcal{N} (x \vert \mu, \Sigma)) = - \frac{1}{2} (x - \mu)^T \Sigma ^{-1} (x - \mu) - \log(\sqrt{(2\pi)^k \vert \Sigma \vert })
$$

## Gradient of Multivariate Gaussian Density Function

Let: 

$$
C = \frac{1}{\sqrt{(2\pi)^k \vert \Sigma \vert}}
$$ 

By the chain rule, we have:

$$
\nabla_x \mathcal{N} (x \vert \mu, \Sigma) = C \exp(- \frac{1}{2} (x - \mu)^T \Sigma ^{-1} (x - \mu)) \nabla_x (- \frac{1}{2} (x - \mu)^T \Sigma ^{-1} (x - \mu))
$$

Let:

$$
f(x) = (x - \mu)^T \Sigma^{-1} (x - \mu)
$$


Taking the differential of both side and product rule, we have:

$$
\begin{aligned}
df &= d((x - \mu)^T \Sigma^{-1} (x - \mu)) \\
   &= (d(x - \mu))^T \Sigma^{-1} (x-\mu) + (x - \mu)^T \Sigma^{-1} d(x-\mu) \\
   &= (x-\mu)^T {\Sigma^{-1}}^T d(x-\mu) + (x - \mu)^T \Sigma^{-1} d(x-\mu) \\
   &= (x-\mu)^T ({\Sigma^{-1}}^T+\Sigma^{-1})d(x-\mu) \\
   &= (\nabla_x f(x))^T d(x-\mu) \\
   \implies \nabla_x f(x) &= (x-\mu)^T (\Sigma^{-1}+{\Sigma^{-1}}^T)
\end{aligned}
$$

The covariance matrix is symmetric. Thus:

$$
\nabla_x (- \frac{1}{2} f(x)) = - \frac{1}{2} (x-\mu)^T (\Sigma^{-1}+{\Sigma^{-1}}) = (x-\mu)^T \Sigma^{-1}
$$

Finally, the gradient of multivariate Gaussian density function is:

$$
\nabla_x \mathcal{N} (x \vert \mu, \Sigma) = C \exp(- \frac{1}{2} (x - \mu)^T \Sigma ^{-1} (x - \mu)) \times (\Sigma^{-1} (x-\mu)) = \frac{\exp(- \frac{1}{2} (x - \mu)^T \Sigma ^{-1} (x - \mu)) \times (\Sigma^{-1} (x-\mu))}{\sqrt{(2\pi)^k \vert \Sigma \vert}}
$$

## Gradient of Multivariate Gaussian Log Density Function/Score Fuction

$$
\nabla_x \log(\mathcal{N} (x \vert \mu, \Sigma)) = - \Sigma^{-1} (x-\mu)
$$

# References:

[Matrix Calculus](https://github.com/LynnHo/Matrix-Calculus-Tutorial)

