pub fn normalize_bytes<const L: usize>(bytes: &mut [u8; L], target: f64) {
    let current: f64 = bytes.iter().map(|byte| *byte as f64).sum();
    let scalar = target / current;
    for i in 0..L {
        bytes[i] = (bytes[i] as f64 * scalar) as u8;
    }
}
