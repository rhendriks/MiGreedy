pub const FIBER_RI: f32 = 1.52;
pub const SPEED_OF_LIGHT: f32 = 299792.458; // km/s
pub const EARTH_RADIUS_KM: f32 = 6371.0;

/// Haversine formula to calculate the great-circle distance between two points
/// given their latitude and longitude in radians.
/// Returns distance in kilometers.
#[inline(always)]
pub fn haversine_distance(lat1: f32, lon1: f32, lat2: f32, lon2: f32) -> f32 {
    let dlat = lat2 - lat1;
    let dlon = lon2 - lon1;

    let a = (dlat / 2.0).sin().powi(2) + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());

    EARTH_RADIUS_KM * c
}

/// Batch haversine: compute distances from one point to many points.
/// Uses SoA (struct-of-arrays) layout so LLVM can auto-vectorize with -O3 + LTO.
/// Writes results into the provided `out` slice (must be same length as input slices).
#[inline(never)] // prevent inlining so LLVM vectorizes the tight loop
pub fn haversine_batch(
    lat1: f32,
    lon1: f32,
    lats2: &[f32],
    lons2: &[f32],
    out: &mut [f32],
) {
    debug_assert_eq!(lats2.len(), lons2.len());
    debug_assert_eq!(lats2.len(), out.len());

    let cos_lat1 = lat1.cos();
    for i in 0..lats2.len() {
        let dlat = lats2[i] - lat1;
        let dlon = lons2[i] - lon1;
        let sin_dlat = (dlat * 0.5).sin();
        let sin_dlon = (dlon * 0.5).sin();
        let a = sin_dlat * sin_dlat + cos_lat1 * lats2[i].cos() * sin_dlon * sin_dlon;
        let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());
        out[i] = EARTH_RADIUS_KM * c;
    }
}
