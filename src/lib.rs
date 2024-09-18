#![allow(incomplete_features)]
#![feature(generic_arg_infer)]
#![feature(generic_const_exprs)]

use std::array::from_fn;

pub type F = f64;

fn search_sorted<I: Ord, const N: usize>(
    v: &[I; N],
    v_ord: &[usize; N],
    tmp: &[I; N * 2],
) -> [usize; N * 2]
where
    [(); N * 2]:,
{
    from_fn(|i| v_ord.iter().position(|&idx| v[idx] > tmp[i]).unwrap_or(N))
}

fn compute_cdf<const N: usize>(w: &[F; N], ord: &[usize; N], cdf_idx: &[usize; N * 2]) -> [F; N * 2]
where
    [(); N + 1]:,
    [(); N * 2]:,
{
    let ord_w = ord.map(|i| w[i]);

    let mut sorted_cdf = [0.; N + 1];
    for (i, w) in ord_w.into_iter().enumerate() {
        sorted_cdf[i + 1] = w + sorted_cdf[i];
    }
    let sum = sorted_cdf.last().unwrap();
    cdf_idx.map(|i| sorted_cdf[i] / sum)
}

fn dist<const N: usize>(cdf_a: &[F; N * 2], cdf_b: &[F; N * 2], deltas: &[F; N * 2 - 1]) -> F
where
    [(); N * 2]:,
    [(); N * 2 - 1]:,
{
    (0..deltas.len())
        .map(|i| (cdf_a[i] - cdf_b[i]).abs() * deltas[i])
        .sum::<F>()
}

/// A swizzled wasserstein distance for finite discrete distributions.
pub fn wasserstein<I: Ord + Copy, const N: usize>(
    a: &[F; N],
    a_idxs: &[I; N],
    b: &[F; N],
    b_idxs: &[I; N],
    idx_dist: impl Fn(I, I) -> F,
) -> F
where
    [(); N + 1]:,
    [(); N * 2]:,
    [(); N * 2 - 1]:,
{
    let mut tmp: [I; N * 2] = from_fn(|i| if i < N { a_idxs[i] } else { b_idxs[i - N] });
    tmp.sort_unstable();

    let mut delta: [F; _] = [0.; N * 2 - 1];
    for i in 0..tmp.len() - 1 {
        delta[i] = idx_dist(tmp[i + 1], tmp[i]);
    }

    let mut a_ord: [usize; N] = from_fn(|i| i);
    a_ord.sort_unstable_by_key(|&i| a_idxs[i]);
    let cdf_idx_a = search_sorted(a_idxs, &a_ord, &tmp);
    let cdf_a = compute_cdf(a, &a_ord, &cdf_idx_a);

    let mut b_ord: [usize; N] = from_fn(|i| i);
    b_ord.sort_unstable_by_key(|&i| b_idxs[i]);
    let cdf_idx_b = search_sorted(b_idxs, &b_ord, &tmp);
    let cdf_b = compute_cdf(b, &b_ord, &cdf_idx_b);

    dist(&cdf_a, &cdf_b, &delta)
}

#[test]
fn test_wasserstein() {
    let w = wasserstein(
        &[0.25; 4],
        &[0, 1, 2, 3],
        &[0.25; 4],
        &[0, 1, 2, 3],
        |l, r| (l - r) as F,
    );
    assert_eq!(w, 0.);

    let w = wasserstein(&[1.], &[0], &[1.], &[1], |l, r| (l - r) as F);
    assert_eq!(w, 1.);

    let w = wasserstein(
        &[3., 1., 1., 1.],
        &[0, 1, 2, 3],
        &[4., 1., 1., 1.],
        &[0, 2, 3, 5],
        |l, r| (l - r) as F,
    );
    assert!((w - 0.571428).abs() < 1e-3, "{w}");
}

#[test]
fn test_binary_wasserstein_distance() {
    let w = wasserstein(
        &[1.],
        &[0],
        &[1.],
        &[100],
        |l, r| if l == r { 0. } else { 1. },
    );
    assert_eq!(w, 1.);

    let w = wasserstein(
        &[7.],
        &[0],
        &[7.],
        &[100],
        |l, r| if l == r { 0. } else { 1. },
    );
    assert_eq!(w, 1.);
}
