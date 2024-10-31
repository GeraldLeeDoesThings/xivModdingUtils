use std::path::Path;

use crate::data::{model_modifiers::ModelImportOptions, tt_model::TexToolsModel};


#[repr(u8)]
pub enum EMeshFlags1 {
    ShadowDisabled = 0x01,
    LightShadowDisabled = 0x02,
    WavingAnimationDisabled = 0x04,
    LightingReflectionEnabled = 0x08,
    Unknown10 = 0x10,
    RainOcclusionEnabled = 0x20,
    SnowOcclusionEnabled = 0x40,
    DustOcclusionEnabled = 0x80,
}

pub fn load_external_model(path: Path) -> Error<TexToolsModel, &'static str> {
    let options = ModelImportOptions::new();
    let suffix = path.extension().ok_or("Path has no extension!")?;
    if suffix.eq("db") {
        todo!()
    }
}
