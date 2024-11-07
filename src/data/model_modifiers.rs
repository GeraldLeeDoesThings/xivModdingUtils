use super::{races::XivRace, textools_model::TexToolsModel};

pub struct ModelImportOptions {
    pub copy_attributes: bool,
    pub copy_materials: bool,
    pub shift_import_uv: bool,
    pub clone_uv2: bool,
    pub auto_scale: bool,
    pub use_imported_tangents: bool,
    pub source_race: XivRace,
    pub target_race: XivRace,
    pub reference_item: (), // TODO: Implement
    pub validate_materials: bool,
    pub source_application: Option<String>,
    pub clear_empty_mesh_data: bool,
    pub auto_assign_heels: bool,
    pub logging_function: (),      // TODO: Implement
    pub intermediary_function: (), // TODO: Implement
}

impl ModelImportOptions {
    pub fn new() -> ModelImportOptions {
        ModelImportOptions {
            copy_attributes: true,
            copy_materials: true,
            shift_import_uv: true,
            clone_uv2: false,
            auto_scale: true,
            use_imported_tangents: false,
            source_race: XivRace::All_Races,
            target_race: XivRace::All_Races,
            reference_item: (),
            validate_materials: true,
            source_application: Some("Unknown".to_string()),
            clear_empty_mesh_data: false,
            auto_assign_heels: true,
            logging_function: (),
            intermediary_function: (),
        }
    }
}
