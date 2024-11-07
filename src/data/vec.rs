use std::ops::{Add, Div, Index, IndexMut, Mul, MulAssign, Sub};

use num_traits::Zero;

macro_rules! impl_common_vec {
    ($name:ty) => {
        impl<T: Sized> Index<usize> for $name {
            type Output = T;
        
            fn index(&self, index: usize) -> &Self::Output {
                &self.data[index]
            }
        }
        
        impl<T: Sized> IndexMut<usize> for $name {
            fn index_mut(&mut self, index: usize) -> &mut Self::Output {
                &mut self.data[index]
            }
        }
    };
}

#[derive(Debug, Clone, Copy)]
pub struct Vec3<T: Sized> {
    data: [T; 3],
}

impl<T: Sized + Copy + Add<Output = O>, O> Add for Vec3<T> {
    type Output = Vec3<O>;

    fn add(self, rhs: Self) -> Self::Output {
        Vec3 {
            data: [
                self.data[0] + rhs.data[0],
                self.data[1] + rhs.data[1],
                self.data[2] + rhs.data[2],
            ],
        }
    }
}

impl<T: Sized + Copy + Div<T, Output = O>, O> Div<T> for &Vec3<T> {
    type Output = Vec3<O>;

    fn div(self, rhs: T) -> Self::Output {
        Vec3::new(
            self[0] / rhs,
            self[1] / rhs,
            self[2] / rhs
        )
    }
}

impl<T: Sized + Copy + Mul<T, Output = T>> MulAssign<T> for Vec3<T> {
    
    fn mul_assign(&mut self, rhs: T) {
        for i in 0..3 {
            self.data[i] = self.data[i] * rhs;
        }
    }
}

impl<T: PartialEq> PartialEq for Vec3<T> {
    fn eq(&self, other: &Self) -> bool {
        self.data[0] == other.data[0] && self.data[1] == other.data[1] && self.data[2] == other.data[2]
    }
}

impl<T: Eq> Eq for Vec3<T> {}

impl<T> Vec3<T> {
    pub fn new(x: T, y: T, z: T) -> Vec3<T> {
        Vec3 { data: [x, y, z] }
    }
}

impl<T: Copy + PartialEq + Zero> Zero for Vec3<T> {
    fn zero() -> Self {
        Vec3::new(T::zero(), T::zero(), T::zero())
    }

    fn is_zero(&self) -> bool {
        *self == Self::zero()
    }
}

impl<T: Copy + Mul<T, Output = T> + Sub<Output = T>> Vec3<T> {
    pub fn cross(&self, other: &Self) -> Vec3<T> {
        Vec3::new(
            self[1] * other[2] - self[2] * other[1],
            self[2] * other[0] - self[0] * other[2],
            self[0] * other[1] - self[1] * other[0],
        )
    }
}

impl<T: Add<T, Output = T> + Copy + Mul<T, Output = T>> Vec3<T> {
    pub fn dot(&self, other: &Self) -> T {
        self[0] * other[0] + self[1] * other[1] + self[2] * other[2]
    }
}

impl Vec3<f32> {
    pub fn length(&self) -> f32 {
        self.dot(self).sqrt()
    }

    pub fn normalize(&self) -> Vec3<f32> {
        self / self.length()
    }
}

impl_common_vec!(Vec3<T>);

#[derive(Debug, Clone, Copy)]
pub struct Vec2<T: Sized> {
    data: [T; 2],
}

impl<T: Sized + Copy + Add<Output = O>, O> Add for Vec2<T> {
    type Output = Vec2<O>;

    fn add(self, rhs: Self) -> Self::Output {
        Vec2 {
            data: [self.data[0] + rhs.data[0], self.data[1] + rhs.data[1]],
        }
    }
}

impl<T> Vec2<T> {
    pub fn new(x: T, y: T) -> Vec2<T> {
        Vec2 { data: [x, y] }
    }
}

impl<T: Copy + PartialEq + Zero> Zero for Vec2<T> {
    fn zero() -> Self {
        Vec2::new(T::zero(), T::zero())
    }

    fn is_zero(&self) -> bool {
        *self == Self::zero()
    }
}

impl<T: PartialEq> PartialEq for Vec2<T> {
    fn eq(&self, other: &Self) -> bool {
        self.data[0] == other.data[0] && self.data[1] == other.data[1]
    }
}

impl<T: Eq> Eq for Vec2<T> {}

impl_common_vec!(Vec2<T>);