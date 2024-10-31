use std::ops::{Add, Index, IndexMut};

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

impl<T> Vec3<T> {
    pub fn new(x: T, y: T, z: T) -> Vec3<T> {
        Vec3 { data: [x, y, z] }
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

impl_common_vec!(Vec2<T>);