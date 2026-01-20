use std::cmp::Ordering;

use num_traits::{Float, FromPrimitive, NumCast, ToPrimitive, Zero};

use crate::{RoundingMode, SciDecimal, SciFloat, SciNum};

impl FromPrimitive for SciDecimal {
    #[inline]
    fn from_i64(n: i64) -> Option<Self> {
        if n > Self::MAX_SIGNIFICAND_SIGNED {
            None
        } else {
            Some(Self::new(n, 0))
        }
    }

    #[inline]
    fn from_u64(n: u64) -> Option<Self> {
        if n > Self::MAX_SIGNIFICAND {
            None
        } else {
            Some(Self::new(n as i64, 0))
        }
    }

    fn from_f64(n: f64) -> Option<Self> {
        Some(<SciDecimal as From<f64>>::from(n))
    }
}

impl ToPrimitive for SciDecimal {
    fn to_i64(&self) -> Option<i64> {
        if self.is_infinite() {
            return None
        }
        match self.precision().cmp(&0) {
            Ordering::Less => {
                // Significand is guaranteed to not be larger than 10^16 - 1 and
                // therefore so is the resulting number
                Some(self.round_precision(0, RoundingMode::HalfUp).significand_signed())
            },
            Ordering::Equal => Some(self.significand_signed()),
            Ordering::Greater => {
                self.significand_signed().checked_mul(10_i64.pow(self.exponent() as u32))
            },
        }
    }

    #[inline]
    fn to_u64(&self) -> Option<u64> {
        if self.is_sign_negative() {
            if self.is_zero() {
                return Some(0)
            } else {
                return None
            }
        }
        self.to_i64().map(|n| n as u64)
    }
}

impl NumCast for SciDecimal {
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
