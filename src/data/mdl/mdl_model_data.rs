use super::mdl::EMeshFlags1;

pub struct MdlModelData {
    pub radius: f32,
    mesh_count: u16,
    attribute_count: u16,
    mesh_part_count: u16,
    material_count: u16,
    bone_count: u16,
    bone_set_count: u16,
    shape_count: u16,
    shape_part_count: u16,
    shape_data_count: u16,
    lod_count: u8,
    flags1: EMeshFlags1,
    // TODO: The rest
}
