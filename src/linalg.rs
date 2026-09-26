pub(crate) fn inverse3(a: [[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let a00 = a[0][0];
    let a01 = a[0][1];
    let a02 = a[0][2];
    let a10 = a[1][0];
    let a11 = a[1][1];
    let a12 = a[1][2];
    let a20 = a[2][0];
    let a21 = a[2][1];
    let a22 = a[2][2];

    let c00 = a11 * a22 - a12 * a21;
    let c01 = -(a10 * a22 - a12 * a20);
    let c02 = a10 * a21 - a11 * a20;

    let c10 = -(a01 * a22 - a02 * a21);
    let c11 = a00 * a22 - a02 * a20;
    let c12 = -(a00 * a21 - a01 * a20);

    let c20 = a01 * a12 - a02 * a11;
    let c21 = -(a00 * a12 - a02 * a10);
    let c22 = a00 * a11 - a01 * a10;

    let det = a00 * c00 + a01 * c01 + a02 * c02;

    if det.abs() < 1e-15 {
        return None;
    }

    let inv = 1.0 / det;

    Some([
        [c00 * inv, c10 * inv, c20 * inv],
        [c01 * inv, c11 * inv, c21 * inv],
        [c02 * inv, c12 * inv, c22 * inv],
    ])
}

pub(crate) fn percentile(values: &[f64], p: f64) -> Option<f64> {
    if values.is_empty() || !(0.0..=100.0).contains(&p) {
        return None;
    }

    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);

    let pos = p / 100.0 * (v.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;

    if lo == hi {
        Some(v[lo])
    } else {
        let weight = pos - lo as f64;
        Some(v[lo] * (1.0 - weight) + v[hi] * weight)
    }
}

pub(crate) fn covariance3(samples: &[[f64; 3]]) -> [[f64; 3]; 3] {
    let n = samples.len() as f64;

    let mut mean = [0.0; 3];

    for x in samples {
        mean[0] += x[0];
        mean[1] += x[1];
        mean[2] += x[2];
    }

    mean[0] /= n;
    mean[1] /= n;
    mean[2] /= n;

    let mut cov = [[0.0; 3]; 3];

    for x in samples {
        let d = [x[0] - mean[0], x[1] - mean[1], x[2] - mean[2]];

        for i in 0..3 {
            for j in 0..3 {
                cov[i][j] += d[i] * d[j];
            }
        }
    }

    let divisor = (n - 1.0).max(1.0);

    for row in &mut cov {
        for x in row {
            *x /= divisor;
        }
    }

    cov
}

pub(crate) fn symmetric_eigen3(mut a: [[f64; 3]; 3]) -> Option<([f64; 3], [[f64; 3]; 3])> {
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    for _ in 0..32 {
        let mut p = 0;
        let mut q = 1;

        if a[0][2].abs() > a[p][q].abs() {
            p = 0;
            q = 2;
        }

        if a[1][2].abs() > a[p][q].abs() {
            p = 1;
            q = 2;
        }

        if a[p][q].abs() < 1e-14 {
            break;
        }

        // angle that zeroes a[p][q]: tan(2θ) = 2·a_pq / (a_qq − a_pp)
        let theta = 0.5 * (2.0 * a[p][q]).atan2(a[q][q] - a[p][p]);

        let c = theta.cos();
        let s = theta.sin();

        for k in 0..3 {
            let apk = a[p][k];
            let aqk = a[q][k];

            a[p][k] = apk * c - aqk * s;
            a[q][k] = apk * s + aqk * c;
        }

        for k in 0..3 {
            let akp = a[k][p];
            let akq = a[k][q];

            a[k][p] = akp * c - akq * s;
            a[k][q] = akp * s + akq * c;
        }

        for k in 0..3 {
            let vkp = v[k][p];
            let vkq = v[k][q];

            v[k][p] = vkp * c - vkq * s;
            v[k][q] = vkp * s + vkq * c;
        }
    }

    // not converged
    let off = a[0][1].abs().max(a[0][2].abs()).max(a[1][2].abs());
    if off > 1e-10 {
        return None;
    }

    Some(([a[0][0], a[1][1], a[2][2]], v))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eigen_decomposes_a_known_matrix() {
        // eigenvalues 1, 3, 4
        let a = [[2.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 4.0]];
        let a_copy = a;
        let (values, vectors) = symmetric_eigen3(a).expect("converges");

        let mut sorted = values;
        sorted.sort_by(f64::total_cmp);
        assert!((sorted[0] - 1.0).abs() < 1e-12);
        assert!((sorted[1] - 3.0).abs() < 1e-12);
        assert!((sorted[2] - 4.0).abs() < 1e-12);

        // A·v = λ·v for every column
        for k in 0..3 {
            for i in 0..3 {
                let av: f64 = (0..3).map(|j| a_copy[i][j] * vectors[j][k]).sum();
                assert!((av - values[k] * vectors[i][k]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn percentile_matches_numpy_linear() {
        // np.percentile([1, 2, 3, 4], 30) == 1.9
        assert!((percentile(&[4.0, 1.0, 3.0, 2.0], 30.0).unwrap() - 1.9).abs() < 1e-12);
    }
}
