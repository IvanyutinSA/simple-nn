use std::{
    fmt,
    ops::{
        Index,
        Add,
        Sub,
        Mul,
    },
};

type MatResult<T> = Result<T, MatError>;

#[derive(Debug)]
pub enum MatError {
    OutOfBounds,
    DimensionMismatch,
}

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
