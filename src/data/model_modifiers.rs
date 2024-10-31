use super::races::XivRace;

pub struct ModelImportOptions {
    copy_attributes: bool,
    copy_materials: bool,
    shift_import_uv: bool,
    clone_uv2: bool,
    auto_scale: bool,
    use_imported_tangents: bool,
    source_race: XivRace,
    target_race: XivRace,
    reference_item: (), // TODO: Implement
    validate_materials: bool,
    source_application: Option<String>,
    clear_empty_mesh_data: bool,
    auto_assign_heels: bool,
    logging_function: (),      // TODO: Implement
    intermediary_function: (), // TODO: Implement
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
