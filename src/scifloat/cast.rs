//! Casting methods and trait implementations for [`SciFloat`].

use num_traits::{FromPrimitive, NumCast, ToPrimitive};

use crate::{SciFloat, SciNum};

macro_rules! impl_from_int {
    ($T:ty) => {
        impl From<$T> for SciFloat {
            fn from(t: $T) -> Self {
                Self::new(t.into())
            }
        }
    };
}

impl_from_int!(i8);
impl_from_int!(i16);
impl_from_int!(i32);
impl_from_int!(u8);
impl_from_int!(u16);
impl_from_int!(u32);

impl From<f64> for SciFloat {
    /// Converts an `f64` into a `SciFloat`.
    fn from(n: f64) -> Self {
        Self {
            number: n,
            uncertainty: 0.0,
        }
    }
}

impl From<SciFloat> for f64 {
    #[inline]
    fn from(n: SciFloat) -> Self {
        n.number()
    }
}

impl From<f32> for SciFloat {
    /// Converts an `f32` into a `SciFloat`.
    fn from(n: f32) -> Self {
        Self {
            number: n.into(),
            uncertainty: 0.0,
        }
    }
}

impl FromPrimitive for SciFloat {
    fn from_i64(n: i64) -> Option<Self> {
        f64::from_i64(n).map(|f| f.into())
    }

    fn from_u64(n: u64) -> Option<Self> {
        f64::from_u64(n).map(|f| f.into())
    }

    fn from_f64(n: f64) -> Option<Self> {
        Some(Self::new(n))
    }
}

impl ToPrimitive for SciFloat {
    fn to_i64(&self) -> Option<i64> {
        self.number().to_i64()
    }

    fn to_u64(&self) -> Option<u64> {
        self.number().to_u64()
    }

    fn to_f64(&self) -> Option<f64> {
        Some(self.number())
    }
}

impl NumCast for SciFloat {
    fn from<T: ToPrimitive>(n: T) -> Option<Self> {
        if let Some(f) = n.to_f64() {
            Self::from_f64(f)
        } else if let Some(i) = n.to_i64() {
            Self::from_i64(i)
        } else {
            Self::from_u64(n.to_u64()?)
        }
    }
}
