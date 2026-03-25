# SEMG
## Convolution and Integration

The Symmetric Exponentially Modified Gaussian (SEMG) profile is the
convolution of a Gaussian with a symmetric Laplace (double-exponential)
distribution. It produces lines with exponential wings that are symmetric
about the centre, giving a profile intermediate between a Gaussian and a
Lorentzian. This document derives the closed-form CDF used in
{func}`~unite.line.functions.integrate_gaussianLaplace`.

---

## The Symmetric Laplace Distribution

The Laplace (double-exponential) distribution centred at zero with scale
parameter $b$ has the probability density

$$
L(x; b) = \frac{1}{2b}\,e^{-|x|/b},
$$

with FWHM $= 2b\ln 2$, so in `unite` the scale parameter is
$b = \text{fwhm\_exp} / (2\ln 2)$ and the corresponding rate is
$\lambda = 1/b = 2\ln 2 / \text{fwhm\_exp}$.

The SEMG profile is the convolution of a Gaussian
$\mathcal{N}(\mu, \sigma^2)$ with $L(x; b)$, where
$\sigma = \sigma_\text{tot} = \sqrt{\sigma_g^2 + \sigma_\text{lsf}^2}$
accounts for both the intrinsic Gaussian component (fwhm\_gauss) and the
instrumental LSF (lsf\_fwhm) added in quadrature. When fwhm\_gauss $= 0$
the profile reduces to a pure Laplace, used by the `Laplace` profile class.

---

## CDF via Symmetrisation of the EMG

The one-sided exponentially modified Gaussian (EMG) has a known
closed-form CDF. Denoting the EMG CDF with rate $\lambda > 0$ as
$G(x; \mu, \sigma, \lambda)$, the symmetric profile's CDF is constructed
by averaging the forward and mirrored EMG:

$$
F(x) = \tfrac{1}{2}\bigl[G(x;\mu,\sigma,\lambda)
+ 1 - G(-x;-\mu,\sigma,\lambda)\bigr].
$$

Expanding $G$ in terms of the normal CDF $\Phi$ and exponential
corrections, and combining, the $1/2$ constants and symmetric Gaussian
tails simplify to:

$$
F(x) = \frac{1}{2} + \frac{1}{2}\,\text{erf}\!\left(\frac{\mu-x}{\sqrt{2}\,\sigma}\right)
- \frac{1}{4}\,e^{a^2}\!\left[
e^{2ta}\,\text{erfc}(t+a) - e^{-2ta}\,\text{erfc}(-t+a)
\right],
$$

where

$$
t = \frac{\mu - x}{\sqrt{2}\,\sigma}, \qquad
a = \frac{\lambda\sigma}{\sqrt{2}} = \frac{\sigma_\text{tot}\ln 2}{\text{fwhm\_exp}/\sqrt{2}}.
$$

The pixel integral over $[\lambda_l, \lambda_h]$ is $F(\lambda_h) - F(\lambda_l)$,
and the Gaussian contribution to this difference is $[\text{erf}(t_l) - \text{erf}(t_h)]/2$,
leaving only the exponential correction term to compute.

---

## Numerically Stable Form via erfcx

The expression $e^{a^2+2ta}\,\text{erfc}(t+a)$ is numerically dangerous:
the exponential grows while `erfc` decays, and both can overflow in
64-bit arithmetic when $a + t$ is large. Rewriting using the
scaled complementary error function
$\text{erfcx}(z) = e^{z^2}\,\text{erfc}(z)$:

$$
e^{a^2} \cdot e^{2ta}\,\text{erfc}(t+a)
= e^{(a+t)^2 - t^2}\,\text{erfc}(t+a)
= e^{-t^2}\,\text{erfcx}(t+a),
$$

and similarly for the second term. The final form is:

$$
\boxed{
F(x) = \frac{1}{2} + \frac{1}{2}\,\text{erf}(t)
- \frac{e^{-t^2}}{4}\bigl[\text{erfcx}(t+a) - \text{erfcx}(-t+a)\bigr]
}
$$

with $t = (\mu-x)/(\sqrt{2}\sigma)$. Since $e^{-t^2}$ cancels the
exponential growth of erfcx, this form is accurate across the full range
of $t$ and $a$.

---

## Connection to the Code

In {func}`~unite.line.functions.integrate_gaussianLaplace`, the code
works in the convention $t_c = (x - \mu)/(\sqrt{2}\sigma) = -t$
(measured from the centre outward rather than inward).
Using the odd symmetry of `erf` and the sign structure of erfcx, the
exponential correction can be written as a single function of $|t_c|$:

$$
\text{\_integrandGL}(t_c, a)
= e^{-t_c^2}\bigl[\text{erfcx}(t_c+a) - \text{erfcx}(-t_c+a)\bigr]
$$

which is equivalent to
`sign(t) * exp(a²) * [exp(2|t|a) erfc(|t|+a) - exp(-2|t|a) erfc(a-|t|)]`
in the implementation, where the sign factor handles the $t \to -t$
convention change and the odd symmetry of the correction.

The full pixel integral is:

$$
\frac{\text{erf}(t_h^c) - \text{erf}(t_l^c)}{2}
+ \frac{\text{\_integrandGL}(t_h^c,\,a) - \text{\_integrandGL}(t_l^c,\,a)}{4}
$$

where the $+1/4$ sign (rather than $-1/4$ from the LaTeX) arises from the
sign flip between $t$ and $t_c$. When $a$ is very large (i.e., the
exponential component is much broader than the pixel scale, the
Gaussian limit), the correction is clipped to zero.
