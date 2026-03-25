# Gauss-Hermite
## Convolution and Integration

The Gauss-Hermite (GH) expansion is a standard parametrization for galaxy
emission-line kinematics, introduced by van der Marel & Franx (1993) and
Gerhard (1993). It represents a profile that departs from a pure Gaussian
through higher-order shape corrections, with $h_3$ controlling skewness and
$h_4$ controlling kurtosis. This document derives the closed-form pixel
integral used in {func}`~unite.line.functions.integrate_gaussHermite` and
shows how the LSF convolution reduces to a simple rescaling of the shape
parameters.

---

## The Gauss-Hermite Distribution

Using the **probabilists' Hermite polynomials** $\text{He}_m$, the
normalised GH profile centred at $\mu$ with width $\sigma$ is

$$
\mathcal{L}(x) = \frac{e^{-y^2/2}}{\sigma\sqrt{2\pi}}
\left[ 1 + \sum_{m=3}^{M} \frac{h_m}{\sqrt{m!}}\,\text{He}_m(y) \right],
\quad y = \frac{x - \mu}{\sigma}.
$$

The first few probabilists' Hermite polynomials are

$$
\text{He}_2(y) = y^2 - 1,\quad
\text{He}_3(y) = y^3 - 3y,\quad
\text{He}_4(y) = y^4 - 6y^2 + 3.
$$

The prefactor $1/\sqrt{m!}$ ensures orthonormality with respect to the
standard normal weight $e^{-y^2/2}$, which guarantees that the profile
integrates to 1 for any values of $h_m$. The sum starts at $m = 3$
because $\text{He}_0 = 1$, $\text{He}_1$, and $\text{He}_2$ are
absorbed into the normalisation, mean, and variance of the base Gaussian.

In `unite` the expansion is truncated at $M = 4$, with free parameters
$h_3$ and $h_4$.

---

## Convolution with a Gaussian LSF

The instrumental line spread function (LSF) is modelled as a Gaussian with
width $\sigma_\text{lsf}$. Convolving $\mathcal{L}$ with this kernel:

$$
(\mathcal{L} * G_\text{lsf})(x)
= \int_{-\infty}^{\infty} \mathcal{L}(x')\,G_\text{lsf}(x - x')\,dx',
\quad G_\text{lsf}(x) = \frac{e^{-x^2/(2\sigma_\text{lsf}^2)}}{\sigma_\text{lsf}\sqrt{2\pi}}.
$$

**Step 1 — Gaussian term.** The zeroth-order term convolvs trivially:

$$
(G_\sigma * G_\text{lsf})(x) = G_{\sigma_\text{new}}(x),
\quad \sigma_\text{new} = \sqrt{\sigma^2 + \sigma_\text{lsf}^2}.
$$

**Step 2 — Derivative identity.** The probabilists' Hermite polynomials
satisfy the Rodrigues formula $\text{He}_m(t)\,e^{-t^2/2} = (-1)^m\,d^m e^{-t^2/2}/dt^m$, from which:

$$
\text{He}_m\!\left(\frac{x}{\sigma}\right) G_\sigma(x)
= (-\sigma)^m \frac{d^m G_\sigma}{dx^m}(x).
$$

**Step 3 — Integration by parts.** Applying the identity to each
Hermite correction and integrating by parts $m$ times (boundary terms vanish
since Gaussians decay at $\pm\infty$):

$$
\int G_\sigma(x')\,\text{He}_m\!\left(\frac{x'}{\sigma}\right)
G_\text{lsf}(x-x')\,dx'
= \sigma^m \frac{d^m G_{\sigma_\text{new}}}{dx^m}(x).
$$

**Step 4 — Apply the identity in reverse.** Converting the derivative back
to a Hermite polynomial using the Rodrigues formula:

$$
\sigma^m \frac{d^m G_{\sigma_\text{new}}}{dx^m}(x)
= \left(\frac{\sigma}{\sigma_\text{new}}\right)^m
\text{He}_m(y_\text{new})\,G_{\sigma_\text{new}}(x),
\quad y_\text{new} = \frac{x - \mu}{\sigma_\text{new}}.
$$

**Result.** The convolved distribution is again a GH profile with
the same $\mu$ but wider width $\sigma_\text{new}$ and rescaled shape
parameters:

$$
(\mathcal{L} * G_\text{lsf})(x)
= \frac{e^{-y_\text{new}^2/2}}{\sigma_\text{new}\sqrt{2\pi}}
\left[ 1 + \sum_{m=3}^{M} \frac{h_m'}{\sqrt{m!}}\,
\text{He}_m(y_\text{new}) \right],
$$

$$
h_m' = h_m \left(\frac{\sigma}{\sigma_\text{new}}\right)^m = h_m\,r^m,
\quad r = \frac{\sigma}{\sigma_\text{new}} < 1.
$$

The ratio $r^m < 1$ suppresses higher-order moments: convolution with the
LSF washes out the non-Gaussian character of the line, with stronger
suppression for higher-order terms.

---

## CDF and Pixel Integration

Exact pixel integration requires the cumulative distribution function (CDF)
$F(x) = \int_{-\infty}^x (\mathcal{L} * G_\text{lsf})(x')\,dx'$.

The Gaussian base term integrates directly to the standard normal CDF
$\Phi(y_\text{new})$. For each Hermite correction, the identity
$\int g(y)\,\text{He}_m(y)\,dy = -g(y)\,\text{He}_{m-1}(y) + C$
(the indefinite integral of a Gaussian-weighted Hermite polynomial
reduces to the next-lower-order polynomial times the Gaussian envelope)
yields:

$$
\int_{-\infty}^x G_{\sigma_\text{new}}(x')\,\text{He}_m(y')\,dx'
= -\frac{g(y_\text{new})}{\sqrt{2\pi}}\,\text{He}_{m-1}(y_\text{new}),
\quad g(y) = e^{-y^2/2}.
$$

Collecting all terms:

$$
\boxed{
F(x) = \Phi(y_\text{new})
- \frac{g(y_\text{new})}{\sqrt{2\pi}}
\sum_{m=3}^{M} \frac{h_m'}{\sqrt{m!}}\,\text{He}_{m-1}(y_\text{new})
}
$$

Each correction term in the CDF is a Gaussian envelope $g(y)$ multiplied
by a polynomial of one order lower than the corresponding PDF correction.
The pixel integral over the bin $[\lambda_l, \lambda_h]$ is then simply
$F(\lambda_h) - F(\lambda_l)$.

---

## Connection to the Code

In {func}`~unite.line.functions.integrate_gaussHermite`, the half-variance
coordinate $t = y/\sqrt{2}$ is used throughout so that `erf` can be
called directly. Translating the CDF formula for $m = 3, 4$:

$$
c_3 = \frac{h_3 r^3}{\sqrt{3!}} = \frac{h_3 r^3}{\sqrt{6}},
\quad
c_4 = \frac{h_4 r^4}{\sqrt{4!}} = \frac{h_4 r^4}{\sqrt{24}},
$$

which correspond to `c3 = h3 * r3 / _SQRT6` and
`c4 = h4 * r3 * r / _SQRT24` in the implementation. The antiderivative
at coordinate $y = t\sqrt{2}$ is:

$$
\text{\_integrandGH}(t; c_3, c_4)
= g(y)\bigl[c_3\,\text{He}_2(y) + c_4\,\text{He}_3(y)\bigr]
= e^{-y^2/2}\bigl[c_3(y^2-1) + c_4\,y(y^2-3)\bigr],
$$

and the final pixel integral is:

$$
\frac{\text{erf}(t_h) - \text{erf}(t_l)}{2}
- \frac{1}{\sqrt{2\pi}}
\bigl[\text{\_integrandGH}(t_h) - \text{\_integrandGH}(t_l)\bigr].
$$
