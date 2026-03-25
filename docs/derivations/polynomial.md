# Polynomial
## Convolution with Gaussian LSF

Polynomial continuum forms (`Linear`, `Polynomial`, `Chebyshev`, `Bernstein`)
are analytically convolved with the Gaussian instrumental LSF at evaluation
time. This is possible because a polynomial convolved with a Gaussian is still
a polynomial, with a coefficient transform that can be precomputed. This
document derives that transform, which is implemented in
{func}`~unite.continuum.library._gaussian_convolve_poly`.

---

## Setup

Let the continuum model in some normalised coordinate $x$ be a polynomial

$$
f(x) = \sum_{i=0}^{N} c_i\,x^i,
$$

and let the LSF be a zero-mean Gaussian kernel with standard deviation
$\sigma$:

$$
G(x;\sigma) = \frac{1}{\sigma\sqrt{2\pi}}\,e^{-x^2/(2\sigma^2)}.
$$

The convolved continuum is

$$
(f * G)(x) = \int_{-\infty}^{\infty} f(t)\,G(x-t;\sigma)\,dt.
$$

---

## Key Idea: Linearity and Moments

By linearity the convolution reduces to a sum over monomials:

$$
(f * G)(x) = \sum_{i=0}^{N} c_i \int_{-\infty}^{\infty} t^i\,G(x-t;\sigma)\,dt
= \sum_{i=0}^N c_i\,M_i(x),
$$

where $M_i(x) = \int_{-\infty}^{\infty} t^i\,G(x-t;\sigma)\,dt$
is the $i$-th moment of the Gaussian centred at $x$.

Substituting $u = t - x$:

$$
M_i(x) = \int_{-\infty}^{\infty} (x+u)^i\,G(u;\sigma)\,du.
$$

Expanding via the binomial theorem $(x+u)^i = \sum_{k=0}^i \binom{i}{k} x^{i-k} u^k$:

$$
M_i(x) = \sum_{k=0}^{i} \binom{i}{k} x^{i-k} \underbrace{\int_{-\infty}^{\infty} u^k\,G(u;\sigma)\,du}_{\mu_k},
$$

where $\mu_k$ are the moments of a zero-mean Gaussian.

---

## Gaussian Moments

All odd moments vanish by symmetry. The even moments are:

$$
\mu_k = \begin{cases}
0 & k\text{ odd}, \\
(k-1)!!\;\sigma^k & k\text{ even},
\end{cases}
$$

where $(k-1)!! = 1 \cdot 3 \cdot 5 \cdots (k-1)$ is the double factorial
(with the convention $(-1)!! = 1$, so $\mu_0 = 1$). For example:
$\mu_0 = 1$, $\mu_2 = \sigma^2$, $\mu_4 = 3\sigma^4$, $\mu_6 = 15\sigma^6$.

---

## The Convolved Polynomial

Substituting back and collecting terms by power of $x$, the convolution
is still a polynomial of the same degree $N$:

$$
(f * G)(x) = \sum_{j=0}^{N} c_j^{\text{new}}\,x^j,
$$

with new coefficients

$$
\boxed{
c_j^{\text{new}}
= \sum_{\substack{k=0\\k\text{ even}}}^{N-j}
c_{j+k}\,\binom{j+k}{k}\,(k-1)!!\;\sigma^k.
}
$$

Each output coefficient $c_j^\text{new}$ receives contributions from all
higher-order input coefficients $c_{j+k}$ (for even $k > 0$): the LSF
smears power from sharper features into smoother ones. The $k = 0$ term
is just $c_j$ itself (LSF has no effect on a constant), and odd moments
contribute nothing because $G$ is symmetric.

---

## Using FWHM Instead of $\sigma$

The Gaussian LSF is parametrised by its FWHM in `unite`:

$$
\sigma = \frac{\text{FWHM}}{2\sqrt{2\ln 2}},
$$

so the coefficient transform becomes

$$
c_j^{\text{new}}
= \sum_{\substack{k=0\\k\text{ even}}}^{N-j}
c_{j+k}\,\binom{j+k}{k}\,(k-1)!!
\left(\frac{\text{FWHM}}{2\sqrt{2\ln 2}}\right)^k.
$$

For basis-transformed forms (`Chebyshev`, `Bernstein`), the FWHM is
first rescaled into the normalised coordinate domain before applying
this transform, since the polynomial lives in $[-1,1]$ or $[0,1]$
rather than wavelength space.

---

## Properties and Connection to the Code

The transform has three important properties:

- **Polynomial closure**: $(f * G)$ is a polynomial of the same degree
  as $f$ — no new basis functions are needed.
- **Only even moments contribute**: odd-$k$ terms vanish, so the sum
  steps by 2. This halves the number of non-zero terms for high-degree
  polynomials.
- **FWHM-controlled**: the entire effect of the LSF enters through a
  single scalar $\sigma$, computed from `lsf_fwhm` at each wavelength
  point.

In {func}`~unite.continuum.library._gaussian_convolve_poly`, the
coefficients are stored in NumPy descending-order convention
(index 0 = highest power), so the index mapping is
$c_j \leftrightarrow \text{coeffs}[N-j]$.
The even-moment double factorial is computed iteratively as
$\mu_{2j} = \mu_{2(j-1)} \cdot (2j-1)\sigma^2$, avoiding explicit
factorial arithmetic. The result is a JAX array of convolved
coefficients that can be evaluated with the same polynomial kernel
used for the unconvolved form.
