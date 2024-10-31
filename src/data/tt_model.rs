use std::{
    collections::{HashMap, HashSet},
    default, fmt::Error,
};

use super::{mdl::EMeshFlags1, vec::{Vec2, Vec3}};

const BONE_ARRAY_LENGTH: usize = 8;

#[derive(PartialEq, Eq)]
pub enum StandardEMesh {
    Standard,
    Water,
    Fog,
}

#[derive(PartialEq, Eq)]
pub enum ExtraEMeshType {
    LightShaft,
    Glass,
    MaterialChange,
    CrestChange,
    ExtraUnknown4,
    ExtraUnknown5,
    ExtraUnknown6,
    ExtraUnknown7,
    ExtraUnknown8,
    ExtraUnknown9,
}

#[derive(PartialEq, Eq)]
pub enum OtherEMeshType {
    Shadow,
    TerrainShadow,
}

#[derive(PartialEq, Eq)]
pub enum EMeshType {
    Standard(StandardEMesh),
    Extra(ExtraEMeshType),
    Other(OtherEMeshType),
}

impl EMeshType {
    fn is_extra_mesh(&self) -> bool {
        match self {
            EMeshType::Extra(_) => true,
            _default => false,
        }
    }

    fn use_zero_default_offset(&self) -> bool {
        self.is_extra_mesh() || self == &EMeshType::Other(OtherEMeshType::TerrainShadow)
    }
}

pub struct TexToolsVertex {
    position: Vec3<f32>,

    normal: Vec3<f32>,
    binormal: Vec3<f32>,
    tangent: Vec3<f32>,
    flow_direction: Vec3<f32>,

    handedness: bool,

    uv1: Vec2<f32>,
    uv2: Vec2<f32>,
    uv3: Vec2<f32>,

    vertex_color: [u8; 4],
    vectex_color2: [u8; 4],

    bone_ids: [u8; BONE_ARRAY_LENGTH],
    weights: [u8; BONE_ARRAY_LENGTH],
}

impl TexToolsVertex {
    pub fn new() -> TexToolsVertex {
        TexToolsVertex {
            position: Vec3::new(0.0, 0.0, 0.0),
            normal: Vec3::new(0.0, 0.0, 0.0),
            binormal: Vec3::new(0.0, 0.0, 0.0),
            tangent: Vec3::new(0.0, 0.0, 0.0),
            flow_direction: Vec3::new(0.0, 0.0, 0.0),
            handedness: false,
            uv1: Vec2::new(0.0, 0.0),
            uv2: Vec2::new(0.0, 0.0),
            uv3: Vec2::new(0.0, 0.0),
            vertex_color: [255, 255, 255, 255],
            vectex_color2: [0, 0, 0, 255],
            bone_ids: [0; BONE_ARRAY_LENGTH],
            weights: [0; BONE_ARRAY_LENGTH],
        }
    }

    #[allow(unused)]
    pub fn get_weld_hash(&self) -> isize {
        unimplemented!()
    }

    #[allow(unused)]
    pub fn get_tangent_space_flow(&self) -> Vec3<f32> {
        unimplemented!()
    }

    #[allow(unused)]
    pub fn world_to_tangent(&self, vec: [f32; 3]) -> [f32; 3] {
        unimplemented!()
    }

    #[allow(unused)]
    pub fn tangent_to_world(&self, vec: [f32; 3]) -> [f32; 3] {
        unimplemented!()
    }
}

pub struct TexToolsMeshPart {
    name: Option<String>,
    verticies: Vec<TexToolsVertex>,
    triangle_indices: Vec<usize>,
    attributes: HashSet<String>,
    shape_parts: HashMap<String, TexToolsShapePart>,
}

impl TexToolsMeshPart {
    pub fn new() -> TexToolsMeshPart {
        TexToolsMeshPart {
            name: None,
            verticies: Vec::new(),
            triangle_indices: Vec::new(),
            attributes: HashSet::new(),
            shape_parts: HashMap::new(),
        }
    }

    #[allow(unused)]
    pub fn get_bounding_box(&self) -> [Vec3<f32>; 2] {
        unimplemented!()
    }

    #[allow(unused)]
    pub fn update_shape_data(&mut self) {
        unimplemented!()
    }
}

pub struct TexToolsShapePart {
    name: Option<String>,
    verticies: Vec<TexToolsVertex>,
    vertex_replacements: HashMap<isize, isize>,
}

impl TexToolsShapePart {
    pub fn new() -> TexToolsShapePart {
        TexToolsShapePart {
            name: None,
            verticies: Vec::new(),
            vertex_replacements: HashMap::new(),
        }
    }
}

pub struct TexToolsMeshGroup {
    parts: Vec<TexToolsMeshPart>,
    material: Option<String>,
    name: Option<String>,
    mesh_type: EMeshType,
    bones: Vec<String>,
}

impl TexToolsMeshGroup {
    pub fn new() -> TexToolsMeshGroup {
        TexToolsMeshGroup {
            parts: Vec::new(),
            material: None,
            name: None,
            mesh_type: EMeshType::Standard(StandardEMesh::Standard),
            bones: Vec::new(),
        }
    }

    #[allow(unused)]
    pub fn get_vertex_count(&self) -> usize {
        unimplemented!()
    }

    #[allow(unused)]
    pub fn get_index_count(&self) -> usize {
        unimplemented!()
    }
    
    #[allow(unused)]
    pub fn set_index_at(&mut self, index_id: usize, vertex_id_to_set: usize) {
        unimplemented!()
    }

    #[allow(unused)]
    pub fn set_vertex_at(&mut self, vertex_id: usize, vertex: TexToolsVertex) {
        unimplemented!()
    }

    #[allow(unused)]
    pub fn get_part_realevant_vertex_information(&self, vertex_id: usize) -> (usize, usize) {
        unimplemented!()
    }

    #[allow(unused)]
    pub fn get_owning_part_id_by_index(&self, mesh_relevant_triangle_index: usize) -> usize {
        unimplemented!()
    }

    #[allow(unused)]
    pub fn get_vertex_at(&mut self, id: usize) -> Option<&mut TexToolsVertex> {
        unimplemented!()
    } 

    #[allow(unused)]
    pub fn get_index_at(&self, id: usize) -> usize {
        unimplemented!()
    }

    #[allow(unused)]
    pub fn update_shape_data(&mut self) {
        for part in &mut self.parts {
            part.update_shape_data();
        }
    }

    #[allow(unused)]
    pub fn part_index_offsets(&self) -> Vec<usize> {
        unimplemented!()
    }

    #[allow(unused)]
    pub fn part_vertex_offsets(&self) -> Vec<usize> {
        unimplemented!()
    }

    #[allow(unused)]
    pub fn vertex_count(&self) -> usize {
        unimplemented!()
    }

    #[allow(unused)]
    pub fn index_count(&self) -> usize {
        unimplemented!()
    }

    #[allow(unused)]
    pub fn triangle_indices(&self) -> Vec<usize> {
        unimplemented!()
    }
}

enum UVAddressingSpace {
    SESpace,
    Standard,
}

pub struct TexToolsModel {
    source: String,
    mdl_version: u16,
    uv_state: UVAddressingSpace,
    mesh_groups: Vec<TexToolsMeshGroup>,
    active_shapes: HashSet<String>,
    anisotropic_lighting_enabled: bool,
    flags: EMeshFlags1,
}

impl TexToolsModel {

    pub fn load_from_file() -> Error<TexToolsModel, &'static str> {
        todo!()
    }

}
