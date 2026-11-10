use std::{
    fmt,
    ops::{
        Index,
        Add,
        Sub,
        Mul,
    },
};
use rand;
use rand_distr::Distribution;

type MatResult<T> = Result<T, MatError>;

#[derive(Debug)]
pub enum MatError {
    OutOfBounds,
    DimensionMismatch,
}

#[derive(Clone, Default)]
pub struct Matrix {
    pub inner: Vec<f64>,
    pub m: usize,
    pub n: usize,
}

impl Matrix {
    pub fn fill(m: usize, n: usize, val: f64) -> Self {
        Self {
            inner: vec![val; m*n],
            m,
            n,
        }
    }

    pub fn project_rows(&self, row_ids: &Vec<usize>) -> Matrix {
        if row_ids.iter().max().unwrap() >= &self.m {
            panic!();
        }
        let mut out_mat = Matrix::fill(row_ids.len(), self.n, 0.);
        for i in 0..row_ids.len() {
            for j in 0..self.n {
                out_mat.set(i, j, self.get(row_ids[i], j).unwrap()).unwrap();
            }
        }
        out_mat
    }

    pub fn project_cols(&self, col_ids: &Vec<usize>) -> Matrix {
        self.transpose().project_rows(col_ids).transpose()
    }

    pub fn std(m: usize, n: usize) -> Self {
        let rng = rand::rng();
        let inner = rand_distr::StandardNormal{}.sample_iter(rng)
            .take(m*n)
            .collect();
        Matrix {
            m, n, inner,
        }
    }

    pub fn he_init(n_in: usize, k_out: usize) -> Self {
        let mean = 0.;
        let std_dev = (2./(n_in as f64)).sqrt();
        let mut rng = rand::rng();
        let inner = rand_distr::StandardNormal.sample_iter(&mut rng)
            .take(n_in*k_out)
            .map(|x: f64| mean + x*std_dev)
            .collect();
        Matrix {
            m: n_in,
            n: k_out,
            inner
        }
    }

    pub fn get(&self, i: usize, j: usize) -> MatResult<f64> {
        if self.m <= i || self.n <= j {
            return Err(MatError::OutOfBounds);
        }
        Ok(self.inner[i*self.n+j])
    }

    pub fn set(&mut self, i: usize, j: usize, val: f64) -> MatResult<()> {
        if self.m <= i || self.n <= j {
            return Err(MatError::OutOfBounds);
        }
        self.inner[i*self.n+j] = val;
        Ok(())
    }

    pub fn mul_pointwise(&self, rhs: &Matrix) -> MatResult<Self> {
        if self.m != rhs.m || self.n != rhs.n {
            return Err(MatError::DimensionMismatch);
        }
        let inner = self.inner.iter()
            .zip(&rhs.inner)
            .map(|(a, b)| a*b)
            .collect();
        Ok(Matrix {
            inner, m: self.m, n: self.n
        })
    }

    pub fn apply(&self, f: impl Fn(f64) -> f64) -> Self {
        Matrix {
            m: self.m,
            n: self.n,
            inner: self.inner.iter().map(|&x| f(x)).collect()
        }
    }

    pub fn transpose(&self) -> Self {
        let mut inner = vec![0.;self.m*self.n];
        for i in 0..self.m {
            for j in 0..self.n {
                inner[j*self.m+i] = self.inner[i*self.n+j];
            }
        }
        Matrix {
            m: self.n,
            n: self.m,
            inner
        }
    }

    pub fn broadcast_rows(&self, m: usize) -> Matrix {
        if self.m != 1 {
            panic!();
        }
        Matrix {
            inner: vec![self.inner.clone();m].into_iter().flatten().collect(),
            m: m,
            n: self.n,
        }
    }

    pub fn sum_rows(&self) -> Matrix {
        let mut mat = Matrix::fill(1, self.n, 0.);
        for i in 0..self.m {
            for j in 0..self.n {
                mat.inner[j] += self.inner[i*self.n+j];
            }
        }
        mat
    }
}

impl Add for &Matrix {
    type Output = Matrix;

    fn add(self, rhs: Self) -> Self::Output {
        if self.m != rhs.m || self.n != rhs.n {
            panic!("dimensions fallen");
        }
        let inner = self.inner.iter()
            .zip(&rhs.inner)
            .map(|(a, b)| a+b)
            .collect();
        Matrix {
            inner, m: self.m, n: self.n
        }
    }
}

impl Add<f64> for &Matrix {
    type Output = Matrix;

    fn add(self, rhs: f64) -> Self::Output {
        let inner = self.inner.iter()
            .map(|x| x+rhs)
            .collect();
        Matrix {
            inner, m: self.m, n: self.n
        }
    }
}

impl Add<&Matrix> for f64 {
    type Output = Matrix;

    fn add(self, rhs: &Matrix) -> Self::Output {
        rhs + self
    }
}

impl Add<f64> for Matrix {
    type Output = Matrix;

    fn add(self, rhs: f64) -> Self::Output {
        let inner = self.inner.iter()
            .map(|x| x+rhs)
            .collect();
        Matrix {
            inner, m: self.m, n: self.n
        }
    }
}

impl Add<Matrix> for f64 {
    type Output = Matrix;

    fn add(self, rhs: Matrix) -> Self::Output {
        rhs + self
    }
}

impl Sub for &Matrix {
    type Output = Matrix;

    fn sub(self, rhs: Self) -> Self::Output {
        if self.m != rhs.m || self.n != rhs.n {
            panic!("dimensions fallen");
        }
        let inner = self.inner.iter()
            .zip(&rhs.inner)
            .map(|(a, b)| a-b)
            .collect();
        Matrix {
            inner, m: self.m, n: self.n
        }
    }
}

impl Mul for &Matrix {
    type Output = Matrix;

    // Matrix product
    fn mul(self, rhs: Self) -> Self::Output {
        if self.n != rhs.m {
            panic!("dimensions fallen");
        }
        let m = self.m;
        let p = self.n;
        let n = rhs.n;
        let mut result = Matrix::fill(m, n, 0.);
        for i in 0..m {
            for j in 0..n {
                let mut cij = 0.;
                for k in 0..p {
                    let mik = self.get(i, k).unwrap();
                    let nkj = rhs.get(k, j).unwrap();
                    cij += mik*nkj;
                }
                result.set(i, j, cij).unwrap();
            }
        }
        result
    }
}

impl From<f64> for Matrix {
    fn from(value: f64) -> Self {
        Matrix {m: 1, n: 1, inner: vec![value]}
    }
}

impl From<Vec<Vec<f64>>> for Matrix {
    fn from(value: Vec<Vec<f64>>) -> Self {
        let m = value.len();
        assert!(m != 0);
        let n = value[0].len();
        assert!(value.iter().all(|row| row.len() == n));
        let inner: Vec<f64> = value
            .into_iter()
            .flatten()
            .collect();
        Matrix {
            m, n, inner
        }
    }
}

impl Mul<f64> for &Matrix {
    type Output = Matrix;

    fn mul(self, rhs: f64) -> Matrix {
        Matrix {
            inner: self.inner.iter().map(|x| x*rhs).collect(),
            m: self.m,
            n: self.n
        }
    }
}

impl Mul<&Matrix> for f64 {
    type Output = Matrix;

    fn mul(self, rhs: &Matrix) -> Matrix {
        rhs * self
    }
}

impl fmt::Debug for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Matrix: {}x{}", self.m, self.n)?;
        for i in 0..self.m {
            write!(f, "[")?;
            for j in 0..self.n {
                if j > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{:8.3}", self.get(i, j).unwrap())?;
            }
            write!(f, " ]\n")?;
        }
        Ok(())
    }
}

// impl From<&Matrix> for f64 {
//     fn from(value: &Matrix) -> Self {
//         if value.n != value.m || value.n != 1 {
//             panic!();
//         }
//         value.inner[0]
//     }
// }

// impl Index<usize> for Matrix {
//     type Output = Vec<f64>;
//
//     fn index(&self, i: usize) -> &Self::Output {
//         &self.inner[i]
//     }
// }
