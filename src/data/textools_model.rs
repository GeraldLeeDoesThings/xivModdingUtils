use std::{
    collections::{HashMap, HashSet}, path::Path, str::FromStr
};

use num_traits::Zero;
use rusqlite::{types::FromSql, Connection, OpenFlags, Row};

use super::{
    mdl::mdl::EMeshFlags1, model_modifiers::ModelImportOptions, transform::normalize_bytes, vec::{Vec2, Vec3}
};

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
    StandardMesh(StandardEMesh),
    ExtraMesh(ExtraEMeshType),
    OtherMesh(OtherEMeshType),
}

#[allow(non_upper_case_globals)]
impl EMeshType {
    pub const Standard: EMeshType = EMeshType::StandardMesh(StandardEMesh::Standard);
    pub const Water: EMeshType = EMeshType::StandardMesh(StandardEMesh::Water);
    pub const Fog: EMeshType = EMeshType::StandardMesh(StandardEMesh::Fog);
    pub const LightShaft: EMeshType = EMeshType::ExtraMesh(ExtraEMeshType::LightShaft);
    pub const Glass: EMeshType = EMeshType::ExtraMesh(ExtraEMeshType::Glass);
    pub const MaterialChange: EMeshType = EMeshType::ExtraMesh(ExtraEMeshType::MaterialChange);
    pub const CrestChange: EMeshType = EMeshType::ExtraMesh(ExtraEMeshType::CrestChange);
    pub const ExtraUnknown4: EMeshType = EMeshType::ExtraMesh(ExtraEMeshType::ExtraUnknown4);
    pub const ExtraUnknown5: EMeshType = EMeshType::ExtraMesh(ExtraEMeshType::ExtraUnknown5);
    pub const ExtraUnknown6: EMeshType = EMeshType::ExtraMesh(ExtraEMeshType::ExtraUnknown6);
    pub const ExtraUnknown7: EMeshType = EMeshType::ExtraMesh(ExtraEMeshType::ExtraUnknown7);
    pub const ExtraUnknown8: EMeshType = EMeshType::ExtraMesh(ExtraEMeshType::ExtraUnknown8);
    pub const ExtraUnknown9: EMeshType = EMeshType::ExtraMesh(ExtraEMeshType::ExtraUnknown9);
    pub const Shadow: EMeshType = EMeshType::OtherMesh(OtherEMeshType::Shadow);
    pub const TerrainShadow: EMeshType = EMeshType::OtherMesh(OtherEMeshType::TerrainShadow);
}

impl EMeshType {
    fn is_extra_mesh(&self) -> bool {
        match self {
            EMeshType::ExtraMesh(_) => true,
            _default => false,
        }
    }

    fn use_zero_default_offset(&self) -> bool {
        self.is_extra_mesh() || self == &EMeshType::OtherMesh(OtherEMeshType::TerrainShadow)
    }
}

impl FromStr for EMeshType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "Standard" => Ok(EMeshType::Standard),
            "Water" => Ok(EMeshType::Water),
            "Fog" => Ok(EMeshType::Fog),
            "LightShaft" => Ok(EMeshType::LightShaft),
            "Glass" => Ok(EMeshType::Glass),
            "MaterialChange" => Ok(EMeshType::MaterialChange),
            "CrestChange" => Ok(EMeshType::CrestChange),
            "ExtraUnknown4" => Ok(EMeshType::ExtraUnknown4),
            "ExtraUnknown5" => Ok(EMeshType::ExtraUnknown5),
            "ExtraUnknown6" => Ok(EMeshType::ExtraUnknown6),
            "ExtraUnknown7" => Ok(EMeshType::ExtraUnknown7),
            "ExtraUnknown8" => Ok(EMeshType::ExtraUnknown8),
            "ExtraUnknown9" => Ok(EMeshType::ExtraUnknown9),
            "Shadow" => Ok(EMeshType::Shadow),
            "TerrainShadow" => Ok(EMeshType::TerrainShadow),
            default => Err(format!("{} is not a valid EMeshType!", default)),
        }
    }
}

#[derive(Clone)]
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
    vertex_color2: [u8; 4],

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
            vertex_color2: [0, 0, 0, 255],
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

    pub fn tangent_to_world(&self, vec: &Vec3<f32>) -> Vec3<f32> {
        
        let normal = self.normal.normalize();
        let binormal = self.binormal.normalize();
        let tangent = self.tangent.normalize();

        let mut transposed: [Vec3<f32>; 3] = [Vec3::zero(); 3];
        for i in 0..3 {
            transposed[i] = Vec3::new(
                normal[0], 
                binormal[0], 
                tangent[0],
            );
        }

        Vec3::new(
            transposed[0].dot(vec), 
            transposed[1].dot(vec), 
            transposed[2].dot(vec),
        )
    }

    pub fn clean_weight(&mut self, max_weights: usize) {
        if max_weights == 4 {
            for i in 4..8 {
                self.weights[i] = 0;
            }
        }

        let weight_sum: usize = self.weights.iter().map(|weight| *weight as usize).sum();
        if weight_sum == 255 { return; }
        else if weight_sum == 0 || weight_sum > 500 {
            self.weights[0] = 255;
            for i in 1..8 {
                self.weights[i] = 0;
            }
        }
        normalize_bytes(&mut self.weights, 255.0);
        
        let bone_sum: u8 = self.weights.iter().sum();
        let diff = 255 - bone_sum;
        let (max_index, _) = self.weights.iter().enumerate().max_by_key(|(_index, val)| **val).unwrap();
        assert!(diff < 255 - self.weights[max_index]);
        self.weights[max_index] += diff;
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

    pub fn update_shape_data(&mut self) {
        for shape in self.shape_parts.values_mut() {
            for (from_index, to_index) in &shape.vertex_replacements {
                let base_vertex = &self.verticies[*from_index];
                let shape_position = shape.verticies[*to_index].position.clone();
                shape.verticies[*to_index] = base_vertex.clone();
                shape.verticies[*to_index].position = shape_position;
            }
        }
    }

    pub fn copy_shape_tangents(&mut self) {
        for shape_part in self.shape_parts.values_mut() {
            for (part_vertex_index, shape_vertex_index) in &shape_part.vertex_replacements {
                let shape_vertex = shape_part.verticies.get_mut(*shape_vertex_index).unwrap();
                let part_vertex = &self.verticies[*part_vertex_index];
                shape_vertex.tangent = part_vertex.tangent;
                shape_vertex.binormal = part_vertex.binormal;
                shape_vertex.handedness = part_vertex.handedness;
            }
        }
    }

    pub fn calculate_tangents_from_binormals(&mut self) {
        for vertex in &mut self.verticies {
            let mut tangent = vertex.normal.cross(&vertex.binormal);
            tangent *= if vertex.handedness { -1.0 } else { 1.0 };
            vertex.tangent = tangent;
        }
        self.copy_shape_tangents();
    }
}

pub struct TexToolsShapePart {
    name: Option<String>,
    verticies: Vec<TexToolsVertex>,
    vertex_replacements: HashMap<usize, usize>,
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
            mesh_type: EMeshType::StandardMesh(StandardEMesh::Standard),
            bones: Vec::new(),
        }
    }

    pub fn get_vertex_count(&self) -> usize {
        self.parts.iter().map(|part| part.verticies.len()).sum()
    }

    pub fn get_index_count(&self) -> usize {
        self.parts.iter().map(|part| part.triangle_indices.len()).sum()
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

    pub fn calculate_tangents(&mut self) {
        if self.vertex_count() == 0 || self.index_count() == 0 { return; }
        let mut any_missing = false;
        for part in &self.parts {
            match part.verticies.iter().find(|vertex| vertex.tangent == Vec3::zero() || vertex.binormal == Vec3::zero()) {
                Some(_) => any_missing = true,
                None => (),
            }
        }
        if !any_missing { return; }

        if let Some(_) = self.parts.iter().find_map(|part| part.verticies.iter().find(|vertex| vertex.binormal != Vec3::zero())) {
            // This logic is odd. If any vertex in any part has non-zero binormals, compute tangents for ALL parts from binormals.
            // This assumes that all binormals are present if at least one is, otherwise we risk computing null tangents
            for part in &mut self.parts {
                // Calc tangents from binormals for part (part)
                part.calculate_tangents_from_binormals();
            }
            return;
        }

        // Hope we never get here...
        unimplemented!("Slow path tangent reacalculation for mesh groups is unimplemented.")
    }
}

#[derive(PartialEq, Eq)]
enum UVAddressingSpace {
    SESpace,
    Standard,
}

pub struct TexToolsModel {
    source: String,
    mdl_version: Option<u16>,
    uv_state: UVAddressingSpace,
    mesh_groups: Vec<TexToolsMeshGroup>,
    active_shapes: HashSet<String>,
    anisotropic_lighting_enabled: Option<bool>,
    flags: Option<EMeshFlags1>,
}

impl TexToolsModel {
    pub fn new() -> TexToolsModel {
        TexToolsModel {
            source: "".to_string(),
            mdl_version: None,
            uv_state: UVAddressingSpace::SESpace,
            mesh_groups: Vec::new(),
            active_shapes: HashSet::new(),
            anisotropic_lighting_enabled: None,
            flags: None,
        }
    }

    pub fn load_from_file(file_path: &Path, settings: &ModelImportOptions) -> Result<TexToolsModel, String> {
        let mut model = TexToolsModel::new();
        let conn = Connection::open_with_flags(
            file_path, 
            OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        ).map_err(|err| err.to_string())?;

        // Load mesh groups
        let mut mesh_group_satement = conn.prepare("SELECT mesh, type, name FROM meshes ORDER BY mesh ASC;").map_err(|err| err.to_string())?;
        let mut mesh_group_rows = mesh_group_satement.query([]).map_err(|err| err.to_string())?;
        while let Some(row) = mesh_group_rows.next().map_err(|err| err.to_string())? {
            let mesh_group_num: usize = row.get_unwrap(0);
            while model.mesh_groups.len() <= mesh_group_num {
                model.mesh_groups.push(TexToolsMeshGroup::new());
            }
            let maybe_mesh_group_type: Option<String> = row.get_unwrap(1);
            let mesh_group_type = maybe_mesh_group_type.unwrap_or("".to_string());
            if mesh_group_type.trim().is_empty() {
                model.mesh_groups[mesh_group_num].mesh_type = EMeshType::StandardMesh(StandardEMesh::Standard);
            }
            else {
                model.mesh_groups[mesh_group_num].mesh_type = EMeshType::from_str(&mesh_group_type).map_err(|errmsg| rusqlite::Error::InvalidParameterName(errmsg)).unwrap();
            }
            model.mesh_groups[mesh_group_num].name = row.get_unwrap(2);
        }

        // Load mesh parts
        let mut mesh_part_statement = conn.prepare("SELECT mesh, part, attributes, name FROM parts ORDER BY mesh ASC, part ASC;").map_err(|err| err.to_string())?;
        let mut mesh_part_rows = mesh_part_statement.query([]).map_err(|err| err.to_string())?;
        while let Some(row) = mesh_part_rows.next().map_err(|err| err.to_string())? {
            let mesh_num: usize = row.get_unwrap(0);
            let part_num: usize = row.get_unwrap(1);

            while model.mesh_groups.len() <= mesh_num {
                model.mesh_groups.push(TexToolsMeshGroup::new());
            }

            while model.mesh_groups[mesh_num].parts.len() <= part_num {
                model.mesh_groups[mesh_num].parts.push(TexToolsMeshPart::new());
            }

            // Clear to mimic replacement
            model.mesh_groups[mesh_num].parts[part_num].attributes.clear();

            let attributes: String = row.get_unwrap(2);
            if !attributes.trim().is_empty() {
                for attribute in attributes.split(',') {
                    model.mesh_groups[mesh_num].parts[part_num].attributes.insert(attribute.to_string());
                }
                model.mesh_groups[mesh_num].parts[part_num].name = Some(row.get_unwrap(3));
            }
        }

        // Load bones
        let mut bone_statement = conn.prepare("SELECT mesh, name FROM bones WHERE mesh >= 0 ORDER BY mesh ASC, bone_id ASC;").map_err(|err| err.to_string())?;
        let mut bone_rows = bone_statement.query([]).map_err(|err| err.to_string())?;
        while let Some(row) = bone_rows.next().map_err(|err| err.to_string())? {
            let mesh_id: usize = row.get_unwrap(0);
            model.mesh_groups[mesh_id].bones.push(row.get_unwrap(1));
        }

        let mut vertex_statement = conn.prepare("SELECT * FROM vertices;").map_err(|err| err.to_string())?;
        let mut vertex_column_names: HashMap<String, usize, _> = HashMap::new();
        vertex_statement.column_names().iter().enumerate().for_each(|(index, name)| { vertex_column_names.insert(name.to_string(), index); });
        let mut vertex_rows = vertex_statement.query([]).map_err(|err| err.to_string())?;
        while let Some(row) = vertex_rows.next().map_err(|err| err.to_string())? {
            let mesh_id: usize = row.get(*vertex_column_names.get("mesh").ok_or("Expected column 'mesh'")?).unwrap();
            let part_id: usize = row.get(*vertex_column_names.get("part").ok_or("Expected column 'part'")?).unwrap();
            let mesh_group = &mut model.mesh_groups[mesh_id];
            let part = &mut mesh_group.parts[part_id];
            let mut vertex = TexToolsVertex::new();

            fn get_col_val<T: FromSql>(row: &Row, name: &str, map: &HashMap<String, usize>) -> Result<T, String> {
                row.get(*map.get(name).ok_or(format!("Expected column '{name}'"))?).map_err(|err| err.to_string())
            } 

            vertex.position = Vec3::new(
                get_col_val(row, "position_x", &vertex_column_names)?,
                get_col_val(row, "position_y", &vertex_column_names)?,
                get_col_val(row, "position_z", &vertex_column_names)?,
            );

            vertex.normal = Vec3::new(
                get_col_val(row, "normal_x", &vertex_column_names)?,
                get_col_val(row, "normal_y", &vertex_column_names)?,
                get_col_val(row, "normal_z", &vertex_column_names)?,
            );

            if settings.use_imported_tangents {
                vertex.binormal = Vec3::new(
                    get_col_val(row, "binormal_x", &vertex_column_names)?,
                    get_col_val(row, "binormal_y", &vertex_column_names)?,
                    get_col_val(row, "binormal_z", &vertex_column_names)?,
                );

                vertex.tangent = Vec3::new(
                    get_col_val(row, "tangent_x", &vertex_column_names)?,
                    get_col_val(row, "tangent_y", &vertex_column_names)?,
                    get_col_val(row, "tangent_z", &vertex_column_names)?,
                );
            }

            vertex.vertex_color[0] = (get_col_val::<f32>(row, "color_r", &vertex_column_names)? * 255.0).round() as u8;
            vertex.vertex_color[1] = (get_col_val::<f32>(row, "color_g", &vertex_column_names)? * 255.0).round() as u8;
            vertex.vertex_color[2] = (get_col_val::<f32>(row, "color_b", &vertex_column_names)? * 255.0).round() as u8;
            vertex.vertex_color[3] = (get_col_val::<f32>(row, "color_a", &vertex_column_names)? * 255.0).round() as u8;

            vertex.vertex_color2[0] = (get_col_val::<f32>(row, "color2_r", &vertex_column_names)? * 255.0).round() as u8;
            vertex.vertex_color2[1] = (get_col_val::<f32>(row, "color2_g", &vertex_column_names)? * 255.0).round() as u8;
            vertex.vertex_color2[2] = (get_col_val::<f32>(row, "color2_b", &vertex_column_names)? * 255.0).round() as u8;
            vertex.vertex_color2[3] = (get_col_val::<f32>(row, "color2_a", &vertex_column_names)? * 255.0).round() as u8;

            vertex.uv1 = Vec2::new(
                get_col_val(row, "uv_1_u", &vertex_column_names)?,
                get_col_val(row, "uv_1_v", &vertex_column_names)?,
            );
            vertex.uv2 = Vec2::new(
                get_col_val(row, "uv_2_u", &vertex_column_names)?,
                get_col_val(row, "uv_2_v", &vertex_column_names)?,
            );
            vertex.uv3 = Vec2::new(
                get_col_val(row, "uv_3_u", &vertex_column_names)?,
                get_col_val(row, "uv_3_v", &vertex_column_names)?,
            );

            for id in 0..8 {
                vertex.bone_ids[id] = get_col_val(row, format!("bone_{id}_id").as_str(), &vertex_column_names)?;
                vertex.weights[id] = (get_col_val::<f32>(row, format!("bone_{id}_weight").as_str(), &vertex_column_names)? * 255.0).round() as u8;
            }

            vertex.flow_direction[0] = get_col_val(row, "flow_u", &vertex_column_names)?;
            vertex.flow_direction[1] = get_col_val(row, "flow_v", &vertex_column_names)?;

            if vertex.binormal != Vec3::new(0.0, 0.0, 0.0) {
                let tangent = vertex.normal.cross(&vertex.binormal).normalize();
                let dot = tangent.dot(&vertex.tangent);
                vertex.handedness = dot < 0.5;
            }

            part.verticies.push(vertex);
        }

        // Triangle indicies
        let mut index_statement = conn.prepare("SELECT mesh, part, vertex_id FROM indices;").map_err(|err| err.to_string())?;
        let mut index_rows = index_statement.query([]).map_err(|err| err.to_string())?;
        while let Some(row) = index_rows.next().map_err(|err| err.to_string())? {
            let mesh_group = &mut model.mesh_groups[row.get_unwrap::<_, usize>(0)];
            let part = &mut mesh_group.parts[row.get_unwrap::<_, usize>(1)];
            part.triangle_indices.push(row.get_unwrap(2));
        }

        // Shape verts
        let mut shape_verts_statement = conn.prepare("SELECT shape, mesh, part, vertex_id, position_x, position_y, position_z FROM shape_vertices ORDER BY shape ASC, mesh ASC, part ASC, vertex_id ASC;").map_err(|err| err.to_string())?;
        let mut shape_verts_rows = shape_verts_statement.query([]).map_err(|err| err.to_string())?;
        while let Some(row) = shape_verts_rows.next().map_err(|err| err.to_string())? {
            let shape_name: String = row.get_unwrap(0);
            let mesh_num: usize = row.get_unwrap(1);
            let part_num: usize = row.get_unwrap(2);
            let vertex_id: usize = row.get_unwrap(3);
            let part = &mut model.mesh_groups[mesh_num].parts[part_num];
            let mut vertex = part.verticies[vertex_id].clone();
            vertex.position = Vec3::new(
                row.get_unwrap(4),
                row.get_unwrap(5),
                row.get_unwrap(6),
            );
            if part.verticies[vertex_id].position == vertex.position {
                continue;
            }
            if !part.shape_parts.contains_key(&shape_name) {
                let mut shape_part = TexToolsShapePart::new();
                shape_part.name = Some(shape_name.clone());
                part.shape_parts.insert(shape_name.clone(), shape_part);
            }

            let num_shape_part_verts = part.shape_parts[&shape_name].verticies.len();
            part.shape_parts.get_mut(&shape_name).unwrap().vertex_replacements.insert(vertex_id, num_shape_part_verts);
        }

        model.uv_state = UVAddressingSpace::Standard;
        model.make_import_ready(true);
        model.calculate_tangents()?;
        model.convert_flow_data();
        model.clean_weights();
        Ok(model)
    }

    pub fn save_to_file(file_path: &Path) {
        todo!()
    }

    pub fn make_import_ready(&mut self, shift_uv: bool) {
        if self.uv_state == UVAddressingSpace::SESpace { return; }
        for model in self.mesh_groups.iter_mut() {
            for part in model.parts.iter_mut() {
                for vertex in part.verticies.iter_mut() {
                    vertex.uv1[1] *= -1.0;
                    vertex.uv2[1] *= -1.0;
                    vertex.uv3[1] *= -1.0;

                    if shift_uv {
                        vertex.uv1[1] += 1.0;
                        vertex.uv2[1] += 1.0;
                        vertex.uv3[1] += 1.0;
                    }
                }
            }
        }
        self.update_shape_data();
        self.uv_state = UVAddressingSpace::SESpace;
    }

    pub fn update_shape_data(&mut self) {
        for mesh_group in &mut self.mesh_groups {
            mesh_group.update_shape_data();
        }
    }

    pub fn has_missing_tangent_data(&self) -> bool {
        for mesh_group in &self.mesh_groups {
            for part in &mesh_group.parts {
                let mut has_tangent = false;
                let mut has_binormal = false;
                if part.verticies.len() == 0 { continue; }

                for vertex in &part.verticies {
                    if !has_tangent && vertex.tangent != Vec3::zero() {
                        has_tangent = true;
                    }

                    if !has_binormal && vertex.binormal != Vec3::zero() {
                        has_binormal = true;
                    }

                    if has_tangent && has_binormal { break; }
                }

                if !has_binormal || !has_tangent { return true; }
            }
        }
        false
    }

    pub fn calculate_tangents(&mut self) -> Result<(), String> {
        if !self.has_missing_tangent_data() { return Ok(()); }

        if self.uv_state != UVAddressingSpace::SESpace {
            return Err("Cannot calculate tangents on model when it is not in SE-style UV space.".to_string())
        }

        let mut reset_shapes: Vec<String> = Vec::new();
        self.active_shapes.iter().for_each(|shape_name| reset_shapes.push(shape_name.clone()));
        self.apply_shapes(&mut Vec::new(), true);
        for mesh_group in &mut self.mesh_groups {
            mesh_group.calculate_tangents();
        }

        if reset_shapes.len() > 0 {
            self.apply_shapes(&mut reset_shapes, true);
        }

        Ok(())
    }

    pub fn apply_shapes(&mut self, shapes: &mut Vec<String>, start_clean: bool) {

        let mut needs_update = false;
        if start_clean {
            match self.active_shapes.iter().find(|active_shape| !shapes.contains(&active_shape)) {
                Some(_) => needs_update = true,
                None => (),
            }

            match shapes.iter().find(|shape_name| !self.active_shapes.contains(*shape_name)) {
                Some(_) => needs_update = true,
                None => (),
            }
        }
        else {
            needs_update = true;
        }

        if !needs_update { return; }

        if start_clean {
            shapes.insert(0, "original".to_string());
        }

        for shape_name in shapes.iter().filter(|shape_name| !self.active_shapes.contains(*shape_name)) {
            for mesh_group in &mut self.mesh_groups {
                for part in mesh_group.parts.iter_mut().filter(|part| part.shape_parts.contains_key(shape_name)) {
                    let shape = &part.shape_parts[shape_name];
                    for (to, from) in &shape.vertex_replacements {
                        part.verticies[*to] = shape.verticies[*from].clone();
                    }
                }
            }
        }

        if start_clean {
            self.active_shapes.clear();
        }

        for shape in shapes {
            if shape != "original" {
                self.active_shapes.insert(shape.clone());
            }
        }
    }

    pub fn convert_flow_data(&mut self) {
        for mesh_group in &mut self.mesh_groups {
            for part in &mut mesh_group.parts {
                for vertex in &mut part.verticies {
                    let flow = Vec3::new(vertex.flow_direction[0], vertex.flow_direction[1], 0.0);
                    let world_flow = vertex.tangent_to_world(&flow).normalize();
                    vertex.flow_direction = world_flow;
                }
            }
        }
    }

    pub fn has_weights(&self) -> bool {
        for mesh_group in &self.mesh_groups {
            if mesh_group.bones.len() > 0 {
                return true;
            }
        }
        false
    }

    pub fn clean_weights(&mut self) {
        if !self.has_weights() {
            return;
        }
        let usage = self.get_usage_info();
        for mesh_group in &mut self.mesh_groups {
            for part in &mut mesh_group.parts {
                for vertex in &mut part.verticies {
                    if usage.needs_8_weights {
                        vertex.clean_weight(8);
                    }
                    else {
                        vertex.clean_weight(4);
                    }
                }
            }
        }
    }

    pub fn get_usage_info(&self) -> TexToolsModelUsageInfo {
        let mut usage_info = TexToolsModelUsageInfo::new();
        for mesh_group in &self.mesh_groups {
            for part in &mesh_group.parts {
                for vertex in &part.verticies {
                    if !usage_info.needs_8_weights {
                        if vertex.weights.len() > 4 {
                            for i in 4..vertex.weights.len() {
                                if vertex.weights[i] > 0 || vertex.bone_ids[i] > 0 {
                                    usage_info.needs_8_weights = true;
                                    break;
                                }
                            }
                        }
                    }
                    if usage_info.max_uv < 2 {
                        if vertex.uv2 != Vec2::zero() {
                            usage_info.max_uv = 2;
                        }
                    }

                    if usage_info.max_uv < 3 {
                        if vertex.uv3 != Vec2::zero() {
                            usage_info.max_uv = 3;
                        }
                    }
                    
                    if !usage_info.uses_v_color2 {
                        if vertex.vertex_color2[0] != 0 || vertex.vertex_color2[1] != 0 || vertex.vertex_color2[2] != 0 || vertex.vertex_color2[3] != 255 {
                            usage_info.uses_v_color2 = true;
                        }
                    }
                }
            }
        }
        usage_info
    }
}

struct TexToolsModelUsageInfo {
    uses_v_color2: bool,
    max_uv: usize,
    needs_8_weights: bool,
}

impl TexToolsModelUsageInfo {
    fn new() -> TexToolsModelUsageInfo {
        TexToolsModelUsageInfo { uses_v_color2: false, max_uv: 1, needs_8_weights: false }
    }
}

