## Multivariate Gaussian Density Function

$$
\mathcal{N} (x \vert \mu, \Sigma) = \frac{\exp( - \frac{1}{2} (x - \mu)^T \Sigma ^{-1} (x - \mu) )}{\sqrt{(2\pi)^k \vert \Sigma \vert }}
$$

Where: $\mu:$ mean matrix, $\Sigma: $ covariance matrix, $x: $ data, $k: $ dimensions of data.

## Multivariate Gaussian Log Density Function

$$
\log(\mathcal{N} (x \vert \mu, \Sigma)) = - \frac{1}{2} (x - \mu)^T \Sigma ^{-1} (x - \mu) - \log(\sqrt{(2\pi)^k \vert \Sigma \vert })
$$

## Gradient of Multivariate Gaussian Density Function

Let: $C = \frac{1}{\sqrt{(2\pi)^k \vert \Sigma \vert}}$. By the chain rule, we have:

$$
\nabla_x \mathcal{N} (x \vert \mu, \Sigma) = C \exp(- \frac{1}{2} (x - \mu)^T \Sigma ^{-1} (x - \mu)) \nabla_x (- \frac{1}{2} (x - \mu)^T \Sigma ^{-1} (x - \mu))
$$

Using the identity: $\nabla_a a^T B a = (B + B^T)a$, the covariance matrix is a symmetric matrix (i.e. $\Sigma = \Sigma^T$). We obtain:

$$
\nabla_x (- \frac{1}{2} (x - \mu)^T \Sigma ^{-1} (x - \mu)) = - \frac{1}{2} \times 2 \Sigma^{-1} (x-\mu) = - \Sigma^{-1} (x-\mu)
$$

Thus, the gradient of multivariate Gaussian density function is:

$$
\nabla_x \mathcal{N} (x \vert \mu, \Sigma) = C \exp(- \frac{1}{2} (x - \mu)^T \Sigma ^{-1} (x - \mu)) \times (\Sigma^{-1} (x-\mu)) = \frac{\exp(- \frac{1}{2} (x - \mu)^T \Sigma ^{-1} (x - \mu)) \times (\Sigma^{-1} (x-\mu))}{\sqrt{(2\pi)^k \vert \Sigma \vert}}
$$

## Gradient of Multivariate Gaussian Log Density Function

$$
\nabla_x \log(\mathcal{N} (x \vert \mu, \Sigma)) = - \Sigma^{-1} (x-\mu)
$$
