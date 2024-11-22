#[derive(Clone)]
pub struct SkeletonData {
    pub bone_name: String,
    pub bone_number: isize,
    pub bone_parent: isize,
    pub pose_matrix: Option<[f32; 16]>,
    pub inverse_pose_matrix: Option<[f32; 16]>,
}
