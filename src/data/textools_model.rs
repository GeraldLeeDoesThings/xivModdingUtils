use std::{
    collections::{HashMap, HashSet},
    fs::{remove_file, File},
    ops::Index,
    path::Path,
    str::FromStr,
};

use num_traits::Zero;
use rusqlite::{named_params, types::FromSql, Connection, OpenFlags, Row};

use super::{
    export_db_sql::CREATE_EXPORT_DB_SQL,
    mdl::mdl::EMeshFlags1,
    model_modifiers::ModelImportOptions,
    skeleton_data::SkeletonData,
    transform::normalize_bytes,
    vec::{Vec2, Vec3},
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

impl ToString for EMeshType {
    fn to_string(&self) -> String {
        match self {
            EMeshType::StandardMesh(standard_emesh) => match standard_emesh {
                StandardEMesh::Standard => "Standard",
                StandardEMesh::Water => "Water",
                StandardEMesh::Fog => "Fog",
            },
            EMeshType::ExtraMesh(extra_emesh_type) => match extra_emesh_type {
                ExtraEMeshType::LightShaft => "LightShaft",
                ExtraEMeshType::Glass => "Glass",
                ExtraEMeshType::MaterialChange => "MaterialChange",
                ExtraEMeshType::CrestChange => "CrestChange",
                ExtraEMeshType::ExtraUnknown4 => "ExtraUnknown4",
                ExtraEMeshType::ExtraUnknown5 => "ExtraUnknown5",
                ExtraEMeshType::ExtraUnknown6 => "ExtraUnknown6",
                ExtraEMeshType::ExtraUnknown7 => "ExtraUnknown7",
                ExtraEMeshType::ExtraUnknown8 => "ExtraUnknown8",
                ExtraEMeshType::ExtraUnknown9 => "ExtraUnknown9",
            },
            EMeshType::OtherMesh(other_emesh_type) => match other_emesh_type {
                OtherEMeshType::Shadow => "Shadow",
                OtherEMeshType::TerrainShadow => "TerrainShadow",
            },
        }
        .to_string()
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
            transposed[i] = Vec3::new(normal[0], binormal[0], tangent[0]);
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
        if weight_sum == 255 {
            return;
        } else if weight_sum == 0 || weight_sum > 500 {
            self.weights[0] = 255;
            for i in 1..8 {
                self.weights[i] = 0;
            }
        }
        normalize_bytes(&mut self.weights, 255.0);

        let bone_sum: u8 = self.weights.iter().sum();
        let diff = 255 - bone_sum;
        let (max_index, _) = self
            .weights
            .iter()
            .enumerate()
            .max_by_key(|(_index, val)| **val)
            .unwrap();
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
        self.parts
            .iter()
            .map(|part| part.triangle_indices.len())
            .sum()
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
        if self.vertex_count() == 0 || self.index_count() == 0 {
            return;
        }
        let mut any_missing = false;
        for part in &self.parts {
            match part
                .verticies
                .iter()
                .find(|vertex| vertex.tangent == Vec3::zero() || vertex.binormal == Vec3::zero())
            {
                Some(_) => any_missing = true,
                None => (),
            }
        }
        if !any_missing {
            return;
        }

        if let Some(_) = self.parts.iter().find_map(|part| {
            part.verticies
                .iter()
                .find(|vertex| vertex.binormal != Vec3::zero())
        }) {
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
pub enum UVAddressingSpace {
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

    pub fn get_bones(&self) -> Vec<&String> {
        let mut unique_bones: HashSet<&String> = HashSet::new();
        for mesh_group in &self.mesh_groups {
            for bone in &mesh_group.bones {
                unique_bones.insert(bone);
            }
        }
        let mut result = Vec::from_iter(unique_bones);
        result.sort();
        result
    }

    pub fn get_materials(&self) -> Vec<&String> {
        let mut unique_materials: HashSet<&String> = HashSet::new();
        for mesh_group in &self.mesh_groups {
            if let Some(material) = &mesh_group.material {
                unique_materials.insert(&material);
            }
        }
        let mut result = Vec::from_iter(unique_materials);
        result.sort();
        result
    }

    pub fn has_ffxiv_path(&self) -> bool {
        false // IT BETTER NOT BE
    }

    pub fn get_material_index(&self, group_number: usize) -> u16 {
        if self.mesh_groups.len() <= group_number {
            return 0;
        }
        let mesh_group = &self.mesh_groups[group_number];
        if let Some(mesh_material) = &mesh_group.material {
            self.get_materials()
                .iter()
                .position(|material| *material == mesh_material)
                .unwrap_or(0) as u16
        } else {
            0
        }
    }

    pub fn load_from_file(
        file_path: &Path,
        settings: &ModelImportOptions,
    ) -> Result<TexToolsModel, String> {
        let mut model = TexToolsModel::new();
        let conn = Connection::open_with_flags(
            file_path,
            OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )
        .map_err(|err| err.to_string())?;

        // Load mesh groups
        let mut mesh_group_satement = conn
            .prepare("SELECT mesh, type, name FROM meshes ORDER BY mesh ASC;")
            .map_err(|err| err.to_string())?;
        let mut mesh_group_rows = mesh_group_satement
            .query([])
            .map_err(|err| err.to_string())?;
        while let Some(row) = mesh_group_rows.next().map_err(|err| err.to_string())? {
            let mesh_group_num: usize = row.get_unwrap(0);
            while model.mesh_groups.len() <= mesh_group_num {
                model.mesh_groups.push(TexToolsMeshGroup::new());
            }
            let maybe_mesh_group_type: Option<String> = row.get_unwrap(1);
            let mesh_group_type = maybe_mesh_group_type.unwrap_or("".to_string());
            if mesh_group_type.trim().is_empty() {
                model.mesh_groups[mesh_group_num].mesh_type =
                    EMeshType::StandardMesh(StandardEMesh::Standard);
            } else {
                model.mesh_groups[mesh_group_num].mesh_type = EMeshType::from_str(&mesh_group_type)
                    .map_err(|errmsg| rusqlite::Error::InvalidParameterName(errmsg))
                    .unwrap();
            }
            model.mesh_groups[mesh_group_num].name = row.get_unwrap(2);
        }

        // Load mesh parts
        let mut mesh_part_statement = conn
            .prepare("SELECT mesh, part, attributes, name FROM parts ORDER BY mesh ASC, part ASC;")
            .map_err(|err| err.to_string())?;
        let mut mesh_part_rows = mesh_part_statement
            .query([])
            .map_err(|err| err.to_string())?;
        while let Some(row) = mesh_part_rows.next().map_err(|err| err.to_string())? {
            let mesh_num: usize = row.get_unwrap(0);
            let part_num: usize = row.get_unwrap(1);

            while model.mesh_groups.len() <= mesh_num {
                model.mesh_groups.push(TexToolsMeshGroup::new());
            }

            while model.mesh_groups[mesh_num].parts.len() <= part_num {
                model.mesh_groups[mesh_num]
                    .parts
                    .push(TexToolsMeshPart::new());
            }

            // Clear to mimic replacement
            model.mesh_groups[mesh_num].parts[part_num]
                .attributes
                .clear();

            let attributes: String = row.get_unwrap(2);
            if !attributes.trim().is_empty() {
                for attribute in attributes.split(',') {
                    model.mesh_groups[mesh_num].parts[part_num]
                        .attributes
                        .insert(attribute.to_string());
                }
                model.mesh_groups[mesh_num].parts[part_num].name = Some(row.get_unwrap(3));
            }
        }

        // Load bones
        let mut bone_statement = conn
            .prepare("SELECT mesh, name FROM bones WHERE mesh >= 0 ORDER BY mesh ASC, bone_id ASC;")
            .map_err(|err| err.to_string())?;
        let mut bone_rows = bone_statement.query([]).map_err(|err| err.to_string())?;
        while let Some(row) = bone_rows.next().map_err(|err| err.to_string())? {
            let mesh_id: usize = row.get_unwrap(0);
            model.mesh_groups[mesh_id].bones.push(row.get_unwrap(1));
        }

        let mut vertex_statement = conn
            .prepare("SELECT * FROM vertices;")
            .map_err(|err| err.to_string())?;
        let mut vertex_column_names: HashMap<String, usize, _> = HashMap::new();
        vertex_statement
            .column_names()
            .iter()
            .enumerate()
            .for_each(|(index, name)| {
                vertex_column_names.insert(name.to_string(), index);
            });
        let mut vertex_rows = vertex_statement.query([]).map_err(|err| err.to_string())?;
        while let Some(row) = vertex_rows.next().map_err(|err| err.to_string())? {
            let mesh_id: usize = row
                .get(
                    *vertex_column_names
                        .get("mesh")
                        .ok_or("Expected column 'mesh'")?,
                )
                .unwrap();
            let part_id: usize = row
                .get(
                    *vertex_column_names
                        .get("part")
                        .ok_or("Expected column 'part'")?,
                )
                .unwrap();
            let mesh_group = &mut model.mesh_groups[mesh_id];
            let part = &mut mesh_group.parts[part_id];
            let mut vertex = TexToolsVertex::new();

            fn get_col_val<T: FromSql>(
                row: &Row,
                name: &str,
                map: &HashMap<String, usize>,
            ) -> Result<T, String> {
                row.get(*map.get(name).ok_or(format!("Expected column '{name}'"))?)
                    .map_err(|err| err.to_string())
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

            vertex.vertex_color[0] =
                (get_col_val::<f32>(row, "color_r", &vertex_column_names)? * 255.0).round() as u8;
            vertex.vertex_color[1] =
                (get_col_val::<f32>(row, "color_g", &vertex_column_names)? * 255.0).round() as u8;
            vertex.vertex_color[2] =
                (get_col_val::<f32>(row, "color_b", &vertex_column_names)? * 255.0).round() as u8;
            vertex.vertex_color[3] =
                (get_col_val::<f32>(row, "color_a", &vertex_column_names)? * 255.0).round() as u8;

            vertex.vertex_color2[0] =
                (get_col_val::<f32>(row, "color2_r", &vertex_column_names)? * 255.0).round() as u8;
            vertex.vertex_color2[1] =
                (get_col_val::<f32>(row, "color2_g", &vertex_column_names)? * 255.0).round() as u8;
            vertex.vertex_color2[2] =
                (get_col_val::<f32>(row, "color2_b", &vertex_column_names)? * 255.0).round() as u8;
            vertex.vertex_color2[3] =
                (get_col_val::<f32>(row, "color2_a", &vertex_column_names)? * 255.0).round() as u8;

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
                vertex.bone_ids[id] =
                    get_col_val(row, format!("bone_{id}_id").as_str(), &vertex_column_names)?;
                vertex.weights[id] = (get_col_val::<f32>(
                    row,
                    format!("bone_{id}_weight").as_str(),
                    &vertex_column_names,
                )? * 255.0)
                    .round() as u8;
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
        let mut index_statement = conn
            .prepare("SELECT mesh, part, vertex_id FROM indices;")
            .map_err(|err| err.to_string())?;
        let mut index_rows = index_statement.query([]).map_err(|err| err.to_string())?;
        while let Some(row) = index_rows.next().map_err(|err| err.to_string())? {
            let mesh_group = &mut model.mesh_groups[row.get_unwrap::<_, usize>(0)];
            let part = &mut mesh_group.parts[row.get_unwrap::<_, usize>(1)];
            part.triangle_indices.push(row.get_unwrap(2));
        }

        // Shape verts
        let mut shape_verts_statement = conn.prepare("SELECT shape, mesh, part, vertex_id, position_x, position_y, position_z FROM shape_vertices ORDER BY shape ASC, mesh ASC, part ASC, vertex_id ASC;").map_err(|err| err.to_string())?;
        let mut shape_verts_rows = shape_verts_statement
            .query([])
            .map_err(|err| err.to_string())?;
        while let Some(row) = shape_verts_rows.next().map_err(|err| err.to_string())? {
            let shape_name: String = row.get_unwrap(0);
            let mesh_num: usize = row.get_unwrap(1);
            let part_num: usize = row.get_unwrap(2);
            let vertex_id: usize = row.get_unwrap(3);
            let part = &mut model.mesh_groups[mesh_num].parts[part_num];
            let mut vertex = part.verticies[vertex_id].clone();
            vertex.position = Vec3::new(row.get_unwrap(4), row.get_unwrap(5), row.get_unwrap(6));
            if part.verticies[vertex_id].position == vertex.position {
                continue;
            }
            if !part.shape_parts.contains_key(&shape_name) {
                let mut shape_part = TexToolsShapePart::new();
                shape_part.name = Some(shape_name.clone());
                part.shape_parts.insert(shape_name.clone(), shape_part);
            }

            let num_shape_part_verts = part.shape_parts[&shape_name].verticies.len();
            part.shape_parts
                .get_mut(&shape_name)
                .unwrap()
                .vertex_replacements
                .insert(vertex_id, num_shape_part_verts);
        }

        model.uv_state = UVAddressingSpace::Standard;
        model.shift_uv_space(true, UVAddressingSpace::SESpace);
        model.calculate_tangents()?;
        model.convert_flow_data();
        model.clean_weights();
        Ok(model)
    }

    pub fn save_to_file(&mut self, file_path: &Path) -> Result<(), String> {
        if !file_path.is_file() {
            return Err(format!(
                "Path {} does not indicate a file!",
                file_path.display()
            ));
        }

        let dir = file_path.parent().ok_or(format!(
            "Failed to extract a parent directory from path '{}'",
            file_path.display()
        ))?;
        let texture_dir = dir;

        self.shift_uv_space(true, UVAddressingSpace::Standard);
        let source_str = Path::new(&self.source)
            .file_stem()
            .map(|os_str| os_str.to_str().unwrap())
            .unwrap_or(&self.source);

        let bones = self.get_bones();
        let mut bone_dict: HashMap<String, SkeletonData> = HashMap::new();
        if self.has_ffxiv_path() && bones.len() > 0 {
            // Technically unreachable since has_ffxiv_path is hard coded to false
            unimplemented!("Loading internal ffxiv models is not currently supported.")
        } else if bones.len() > 0 {
            for (index, bone) in bones.iter().enumerate() {
                bone_dict.insert(
                    (**bone).clone(),
                    SkeletonData {
                        bone_name: (**bone).clone(),
                        bone_number: index as isize,
                        bone_parent: 0,
                        pose_matrix: None,
                        inverse_pose_matrix: None,
                    },
                );
            }
        }
        let db_conn = Connection::open(file_path).map_err(|_| {
            format!(
                "Filed to obtain sqlite connection to {}",
                file_path.display()
            )
        })?;
        db_conn
            .execute(&CREATE_EXPORT_DB_SQL, [])
            .map_err(|err| err.to_string())?;

        let mut metadata_query = db_conn
            .prepare("INSERT INTO meta (key, value) VALUES ($key, $value);")
            .map_err(|err| err.to_string())?;
        metadata_query
            .execute(named_params! { "$key": "unit", "$value": "meter" })
            .map_err(|err| err.to_string())?;
        metadata_query
            .execute(named_params! { "$key": "application", "$value": "ffxiv_tt_rust" })
            .map_err(|err| err.to_string())?;
        metadata_query
            .execute(named_params! { "$key": "version", "$value": env!("CARGO_PKG_VERSION") })
            .map_err(|err| err.to_string())?;
        metadata_query
            .execute(named_params! { "$key": "up", "$value": "y" })
            .map_err(|err| err.to_string())?;
        metadata_query
            .execute(named_params! { "$key": "front", "$value": "z" })
            .map_err(|err| err.to_string())?;
        metadata_query
            .execute(named_params! { "$key": "handedness", "$value": "r" })
            .map_err(|err| err.to_string())?;
        // Don't know exactly what to do with this
        metadata_query
            .execute(named_params! { "$key": "for_3ds_max", "$value": "0" })
            .map_err(|err| err.to_string())?;
        metadata_query
            .execute(named_params! { "$key": "root_name", "$value": source_str})
            .map_err(|err| err.to_string())?;

        let mut skeleton_query = db_conn.prepare("INSERT INTO skeleton (name, parent, matrix_0, matrix_1, matrix_2, matrix_3, matrix_4, matrix_5, matrix_6, matrix_7, matrix_8, matrix_9, matrix_10, matrix_11, matrix_12, matrix_13, matrix_14, matrix_15) VALUES ($name, $parent, $matrix_0, $matrix_1, $matrix_2, $matrix_3, $matrix_4, $matrix_5, $matrix_6, $matrix_7, $matrix_8, $matrix_9, $matrix_10, $matrix_11, $matrix_12, $matrix_13, $matrix_14, $matrix_15);").map_err(|err| err.to_string())?;
        for (_, bone_data) in &bone_dict {
            let parent_name = bone_dict
                .iter()
                .find(|(_, pbone_data)| pbone_data.bone_number == bone_data.bone_parent)
                .map(|(pbone_name, pbone_data)| pbone_name);
            skeleton_query
                .execute(named_params! {
                    "$name": bone_data.bone_name,
                    "$parent": parent_name,
                    "$matrix_0": bone_data.pose_matrix.map(|mat| mat[0]),
                    "$matrix_1": bone_data.pose_matrix.map(|mat| mat[1]),
                    "$matrix_2": bone_data.pose_matrix.map(|mat| mat[2]),
                    "$matrix_3": bone_data.pose_matrix.map(|mat| mat[3]),
                    "$matrix_4": bone_data.pose_matrix.map(|mat| mat[4]),
                    "$matrix_5": bone_data.pose_matrix.map(|mat| mat[5]),
                    "$matrix_6": bone_data.pose_matrix.map(|mat| mat[6]),
                    "$matrix_7": bone_data.pose_matrix.map(|mat| mat[7]),
                    "$matrix_8": bone_data.pose_matrix.map(|mat| mat[8]),
                    "$matrix_9": bone_data.pose_matrix.map(|mat| mat[9]),
                    "$matrix_10": bone_data.pose_matrix.map(|mat| mat[10]),
                    "$matrix_11": bone_data.pose_matrix.map(|mat| mat[11]),
                    "$matrix_12": bone_data.pose_matrix.map(|mat| mat[12]),
                    "$matrix_13": bone_data.pose_matrix.map(|mat| mat[13]),
                    "$matrix_14": bone_data.pose_matrix.map(|mat| mat[14]),
                    "$matrix_15": bone_data.pose_matrix.map(|mat| mat[15]),
                })
                .map_err(|err| err.to_string())?;
        }

        db_conn
            .execute(
                "INSERT INTO models (model, name) VALUES ($model, $name);",
                named_params! { "$model": 0, "$name": source_str },
            )
            .map_err(|err| err.to_string())?;

        let mut material_query = db_conn.prepare("INSERT INTO materials (material_id, name, diffuse, normal, specular, opacity, emissive) VALUES ($material_id, $name, $diffuse, $normal, $specular, $opacity, $emissive);").map_err(|err| err.to_string())?;
        for (material_id, material) in self.get_materials().iter().enumerate() {
            let material_filename = Path::new(&material[1..]);
            let mut material_prefix = texture_dir
                .join(material_filename.file_stem().unwrap())
                .to_str()
                .unwrap()
                .to_string();
            material_prefix.push('_');
            let material_suffix = ".png";
            let name = Path::new(&material).file_stem().unwrap().to_str().unwrap();
            material_query
                .execute(named_params! {
                    "material_id": material_id,
                    "name": name,
                    "diffuse": material_prefix.clone() + "d" + material_suffix,
                    "normal": material_prefix.clone() + "n" + material_suffix,
                    "specular": material_prefix.clone() + "s" + material_suffix,
                    "emissive": material_prefix.clone() + "e" + material_suffix,
                    "opacity": material_prefix.clone() + "o" + material_suffix,
                })
                .map_err(|err| err.to_string())?;
        }

        for (mesh_group_id, mesh_group) in self.mesh_groups.iter().enumerate() {
            let mut bone_query = db_conn
                .prepare("INSERT INTO bones (mesh, bone_id, name) VALUES ($mesh, $bone_id, $name);")
                .map_err(|err| err.to_string())?;
            for (bone_id, bone) in mesh_group.bones.iter().enumerate() {
                bone_query
                    .execute(named_params! {
                        "$name": bone,
                        "$bone_id": bone_id,
                        "$mesh": mesh_group_id,
                    })
                    .map_err(|err| err.to_string())?;
            }

            let mut group_query = db_conn.prepare("INSERT INTO meshes (mesh, name, material_id, model, type) VALUES ($mesh, $name, $material_id, $model, $type);").map_err(|err| err.to_string())?;
            group_query
                .execute(named_params! {
                    "$name": mesh_group.name,
                    "$mesh": mesh_group_id,
                    "$model": 0,
                    "$material_id": self.get_material_index(mesh_group_id),
                    "$type": mesh_group.mesh_type.to_string(),

                })
                .map_err(|err| err.to_string())?;

            for (part_id, part) in mesh_group.parts.iter().enumerate() {
                let mut part_query = db_conn.prepare("INSERT INTO parts (mesh, part, name, attributes) VALUES ($mesh, $part, $name, $attributes);").map_err(|err| err.to_string())?;
                part_query
                    .execute(named_params! {
                        "$name": part.name,
                        "$part": part_id,
                        "$mesh": mesh_group_id,
                        "$attributes": itertools::join(&part.attributes, ","),
                    })
                    .map_err(|err| err.to_string())?;

                for (vertex_id, vertex) in part.verticies.iter().enumerate() {
                    let mut vertex_query = db_conn.prepare("INSERT INTO vertices (mesh, part, vertex_id, position_x, position_y, position_z, normal_x, normal_y, normal_z, binormal_x, binormal_y, binormal_z, tangent_x, tangent_y, tangent_z, color_r, color_g, color_b, color_a, color2_r, color2_g, color2_b, color2_a, uv_1_u, uv_1_v, uv_2_u, uv_2_v, bone_1_id, bone_1_weight, bone_2_id, bone_2_weight, bone_3_id, bone_3_weight, bone_4_id, bone_4_weight, bone_5_id, bone_5_weight, bone_6_id, bone_6_weight, bone_7_id, bone_7_weight, bone_8_id, bone_8_weight, uv_3_u, uv_3_v, flow_u, flow_v) values ($mesh, $part, $vertex_id, $position_x, $position_y, $position_z, $normal_x, $normal_y, $normal_z, $binormal_x, $binormal_y, $binormal_z, $tangent_x, $tangent_y, $tangent_z, $color_r, $color_g, $color_b, $color_a, $color2_r, $color2_g, $color2_b, $color2_a, $uv_1_u, $uv_1_v, $uv_2_u, $uv_2_v, $bone_1_id, $bone_1_weight, $bone_2_id, $bone_2_weight, $bone_3_id, $bone_3_weight, $bone_4_id, $bone_4_weight, $bone_5_id, $bone_5_weight, $bone_6_id, $bone_6_weight, $bone_7_id, $bone_7_weight, $bone_8_id, $bone_8_weight, $uv_3_u, $uv_3_v, $flow_u, $flow_v);").map_err(|err| err.to_string())?;
                    let flow = vertex.get_tangent_space_flow();
                    vertex_query
                        .execute(named_params! {
                            "$part": part_id,
                            "$mesh": mesh_group_id,
                            "$vertex_id": vertex_id,
                            "$position_x": vertex.position[0],
                            "$position_y": vertex.position[1],
                            "$position_z": vertex.position[2],
                            "$normal_x": vertex.normal[0],
                            "$normal_y": vertex.normal[1],
                            "$normal_z": vertex.normal[2],
                            "$binormal_x": vertex.binormal[0],
                            "$binormal_y": vertex.binormal[1],
                            "$binormal_z": vertex.binormal[2],
                            "$tangent_x": vertex.tangent[0],
                            "$tangent_y": vertex.tangent[1],
                            "$tangent_z": vertex.tangent[2],
                            "$color_r": vertex.vertex_color[0] as f32 / 255.0f32,
                            "$color_g": vertex.vertex_color[1] as f32 / 255.0f32,
                            "$color_b": vertex.vertex_color[2] as f32 / 255.0f32,
                            "$color_a": vertex.vertex_color[3] as f32 / 255.0f32,
                            "$color2_r": vertex.vertex_color2[0] as f32 / 255.0f32,
                            "$color2_g": vertex.vertex_color2[1] as f32 / 255.0f32,
                            "$color2_b": vertex.vertex_color2[2] as f32 / 255.0f32,
                            "$color2_a": vertex.vertex_color2[3] as f32 / 255.0f32,
                            "$uv_1_u": vertex.uv1[0],
                            "$uv_1_v": vertex.uv1[1],
                            "$uv_2_u": vertex.uv2[0],
                            "$uv_2_v": vertex.uv2[1],
                            "$uv_3_u": vertex.uv3[0],
                            "$uv_3_v": vertex.uv3[1],
                            "$bone_1_id": vertex.bone_ids[0],
                            "$bone_1_weight": vertex.weights[0] as f32 / 255.0f32,
                            "$bone_2_id": vertex.bone_ids[1],
                            "$bone_2_weight": vertex.weights[1] as f32 / 255.0f32,
                            "$bone_3_id": vertex.bone_ids[2],
                            "$bone_3_weight": vertex.weights[2] as f32 / 255.0f32,
                            "$bone_4_id": vertex.bone_ids[3],
                            "$bone_4_weight": vertex.weights[3] as f32 / 255.0f32,
                            "$bone_5_id": vertex.bone_ids[4],
                            "$bone_5_weight": vertex.weights[4] as f32 / 255.0f32,
                            "$bone_6_id": vertex.bone_ids[5],
                            "$bone_6_weight": vertex.weights[5] as f32 / 255.0f32,
                            "$bone_7_id": vertex.bone_ids[6],
                            "$bone_7_weight": vertex.weights[6] as f32 / 255.0f32,
                            "$bone_8_id": vertex.bone_ids[7],
                            "$bone_8_weight": vertex.weights[7] as f32 / 255.0f32,
                            "$flow_u": flow[0],
                            "$flow_v": flow[1],
                        })
                        .map_err(|err| err.to_string())?;
                }

                for (index_id, index) in part.triangle_indices.iter().enumerate() {
                    let mut index_query = db_conn.prepare("INSERT INTO indices (mesh, part, index_id, vertex_id) VALUES ($mesh, $part, $index_id, $vertex_id);").map_err(|err| err.to_string())?;
                    index_query
                        .execute(named_params! {
                            "$part": part_id,
                            "$mesh": mesh_group_id,
                            "$index_id": index_id,
                            "$vertex_id": index,
                        })
                        .map_err(|err| err.to_string())?;
                }

                for (shape_key, shape) in &part.shape_parts {
                    if shape_key.starts_with("shp_") {
                        continue;
                    }
                    let mut index_query = db_conn.prepare("INSERT INTO shape_vertices (mesh, part, shape, vertex_id, position_x, position_y, position_z) VALUES ($mesh, $part, $shape, $vertex_id, $position_x, $position_y, $position_z);").map_err(|err| err.to_string())?;
                    for (original_id, replacement_id) in &shape.vertex_replacements {
                        let vertex = shape
                            .verticies
                            .get(*original_id)
                            .ok_or("Vertex replacement index is out of bounds!")?;
                        index_query
                            .execute(named_params! {
                                "$part": part_id,
                                "$mesh": mesh_group_id,
                                "$shape": shape_key,
                                "$vertex_id": replacement_id,
                                "$position_x": vertex.position[0],
                                "$position_y": vertex.position[1],
                                "$position_z": vertex.position[2],
                            })
                            .map_err(|err| err.to_string())?;
                    }
                }
            }
        }

        Ok(())
    }

    pub fn shift_uv_space(&mut self, shift_uv: bool, target_space: UVAddressingSpace) {
        if self.uv_state == target_space {
            return;
        }
        for mesh_group in &mut self.mesh_groups {
            for part in &mut mesh_group.parts {
                for vertex in &mut part.verticies {
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
        self.uv_state = target_space;
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
                if part.verticies.len() == 0 {
                    continue;
                }

                for vertex in &part.verticies {
                    if !has_tangent && vertex.tangent != Vec3::zero() {
                        has_tangent = true;
                    }

                    if !has_binormal && vertex.binormal != Vec3::zero() {
                        has_binormal = true;
                    }

                    if has_tangent && has_binormal {
                        break;
                    }
                }

                if !has_binormal || !has_tangent {
                    return true;
                }
            }
        }
        false
    }

    pub fn calculate_tangents(&mut self) -> Result<(), String> {
        if !self.has_missing_tangent_data() {
            return Ok(());
        }

        if self.uv_state != UVAddressingSpace::SESpace {
            return Err(
                "Cannot calculate tangents on model when it is not in SE-style UV space."
                    .to_string(),
            );
        }

        let mut reset_shapes: Vec<String> = Vec::new();
        self.active_shapes
            .iter()
            .for_each(|shape_name| reset_shapes.push(shape_name.clone()));
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
            match self
                .active_shapes
                .iter()
                .find(|active_shape| !shapes.contains(&active_shape))
            {
                Some(_) => needs_update = true,
                None => (),
            }

            match shapes
                .iter()
                .find(|shape_name| !self.active_shapes.contains(*shape_name))
            {
                Some(_) => needs_update = true,
                None => (),
            }
        } else {
            needs_update = true;
        }

        if !needs_update {
            return;
        }

        if start_clean {
            shapes.insert(0, "original".to_string());
        }

        for shape_name in shapes
            .iter()
            .filter(|shape_name| !self.active_shapes.contains(*shape_name))
        {
            for mesh_group in &mut self.mesh_groups {
                for part in mesh_group
                    .parts
                    .iter_mut()
                    .filter(|part| part.shape_parts.contains_key(shape_name))
                {
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
                    } else {
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
                        if vertex.vertex_color2[0] != 0
                            || vertex.vertex_color2[1] != 0
                            || vertex.vertex_color2[2] != 0
                            || vertex.vertex_color2[3] != 255
                        {
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
        TexToolsModelUsageInfo {
            uses_v_color2: false,
            max_uv: 1,
            needs_8_weights: false,
        }
    }
}
