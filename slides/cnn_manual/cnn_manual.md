title: Manual Convolution
section: Forward Pass
# Convolution – Forward Computation

Let
- $⇶I$ be an image of size $[H, W, C]$,
~~~
- $⇶K$ be a kernel of size $[k, k, C, O]$,
~~~
- stride be $s$.

~~~
Convolution (or more correctly cross-correlation) is by definition computed as
$$(⇶K \star ⇶I)_{i, j, o} = b_o + ∑_{m=0}^{k-1} ∑_{n=0}^{k-1} ∑_c ⇶I_{i + m, j + n, c} ⇶K_{m, n, c, o}.$$

---
# Convolution – Forward Computation

We denote the result as $⇶R = ⇶K \star ⇶I$, and we start vectorizing it first as
$$⇶R_{i, j\textcolor{lightgray}{, o}} = →b + ∑_{m=0}^{k-1} ∑_{n=0}^{k-1} ∑_c ⇶I_{i + m, j + n, c} ⇶K_{m, n, c\textcolor{lightgray}{, o}},$$

~~~
and finally as
$$⇶R_{i, j\textcolor{lightgray}{, o}} = →b +∑_{m=0}^{k-1} ∑_{n=0}^{k-1} \textcolor{lightgray}{∑_c}
  ⇶I_{i + m, j + n\textcolor{lightgray}{, c}} ⇶K_{m, n\textcolor{lightgray}{, c, o}},$$

where $⇶I_{i + m, j + n}$ is a vector of size $C$ and $⇶K_{m, n}$ a matrix of
size $[C, O]$.

---
# Convolution – Forward Computation

To compute the result quickly, we need to compute it for all image positions in
parallel. Therefore, we rearrange
$$⇶R_{i, j} = →b +∑_{m=0}^{k-1} ∑_{n=0}^{k-1} ⇶I_{i + m, j + n} ⇶K_{m, n}
  = →b +⇶I_{i,j} ⇉K_{0,0} + ⇶I_{i+1,j} ⇉K_{1,0} + ⇶I_{i,j+1} ⇉K_{0,1} + …
  \textrm{~~~~~~as} $$
~~~
- $⇶R ← →b$
- for $0 ≤ m < k$,
  - for $0 ≤ n < k$,
    - $⇶R ← ⇶R + ⇶I_{•, •} ⇶K_{m,n}$

~~~
      Because the output size is $[H-(k-1), W-(k-1)]$, the correct indices are:
    - $⇶R ← ⇶R + ⇶I_{m:m+H-(k-1), n:n+W-(k-1)} ⇶K_{m,n}$

~~~
Finally, for stride $s$, we only modify the image indices to $⇶I_{m:m+H-(k-1):s, n:n+W-(k-1):s}$.

---
section: Backward Pass
# Convolution – Backward Computation

Now assume we got $⇶G = \frac{∂L}{∂⇶R}$, which is of size $\big[\lceil\frac{H-(k-1)}{s}\rceil, \lceil\frac{W-(k-1)}{s}\rceil, O\big]$.

~~~
- $\displaystyle \frac{∂L}{∂→b} = ∑_i ∑_j \textcolor{darkgreen}{\frac{∂⇶R_{i,j}}{∂→b}} ⇶G_{i,j} = ∑_i ∑_j \textcolor{darkgreen}{⇉I} ⇶G_{i,j}.$

~~~
- $\displaystyle \frac{∂L}{∂⇶K_{m,n}} = ∑_i ∑_j \textcolor{darkblue}{\frac{∂⇶R_{i,j}}{∂⇶K_{m,n}}} (⇶G_{i,j})^\T$

~~~
  $\displaystyle \phantom{\frac{∂L}{∂⇶K_{m,n}}} = ∑_i ∑_j \textcolor{darkblue}{⇶I_{si+m,sj+n}} (⇶G_{i,j})^\T$

---
# Convolution – Backward Computation

- Recall that
  $$⇶R_{i, j} = →b +∑_{m=0}^{k-1} ∑_{n=0}^{k-1} ⇶I_{i + m, j + n} ⇶K_{m, n}
  = →b +⇶I_{i,j} ⇉K_{0,0} + ⇶I_{i+1,j} ⇉K_{1,0} + ⇶I_{i,j+1} ⇉K_{0,1} + …$$

  Assuming stride 1 for a while,

  $\displaystyle \frac{∂L}{∂⇶I_{i',j'}} = ∑_i ∑_j \textcolor{darkred}{\frac{∂⇶R_{i,j}}{∂⇶I_{i',j'}}} ⇶G_{i,j}$

~~~
  $\displaystyle \phantom{\frac{∂L}{∂⇶I_{i',j'}}} = \textcolor{darkred}{∑_{m=0}^{k-1} ∑_{n=0}^{k-1} ⇶K_{m,n}} ⇶G_{i,j}$ for $i'=i+m$, $j'=j+n$

~~~
  $\displaystyle \phantom{\frac{∂L}{∂⇶I_{i',j'}}} = \textcolor{darkred}{∑_{m=0}^{k-1} ∑_{n=0}^{k-1} ⇶K_{m,n}} ⇶G_{i'-m,j'-n}$

---
# Convolution – Backward Computation

$$\frac{∂L}{∂⇶I_{i',j'}} = ∑_{m=0}^{k-1} ∑_{n=0}^{k-1} ⇶K_{m,n} ⇶G_{i'-m,j'-n}$$

~~~
We can compute the above analogously to the forward pass of a convolution, but
we must be careful about the indices. Notably, many of the $i'-m$, $j'-n$ are
outside of $⇶G$: the indices can be up to $k-1$ from the left/top edge of $⇶G$,
and they can be up to $k-1$ from the right/bottom edge of $⇶G$, so the easiest
is to pad $⇶G$ with $k-1$ on both sides.

~~~
For completeness, note that the above formulation can be rewritten to a regular
convolution by substituting $m' = k-1-m$ and $n' = k-1-n$:
$$\frac{∂L}{∂⇶I_{i',j'}} = ∑_{m'=0}^{k-1} ∑_{n'=0}^{k-1} ⇶K_{k-1-m',k-1-n'} ⇶G_{i'-(k-1)+m',j'-(k-1)+n'},$$
which is obviously a convolution, but with a point-reflected kernel (i.e., rotated by 180°).

---
section: Stride>1
# Convolution – Backward Computation

Finally, consider a stride $s>1$. During forward pass, we usually keep
only the output values corresponding to the positions where the kernel
was really used, so the output is approximately $s$-times smaller.

~~~
However, we could also keep the output of the original size, but instead
use zero values on the positions where the kernel was not applied. That is
definitely less efficient, but it makes the output size independent on the
stride, so the gradient computation with respect to convolution input then
works for any stride.

![w=65%,h=center](striding_via_zeros.svgz)
