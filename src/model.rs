use rstar::{RTreeObject, AABB};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct Airport {
    pub iata: String,
    pub lat: f32,
    pub lon: f32,
    pub pop: u32,
    pub city: String,
    pub country_code: String,
    pub lat_rad: f32,
    pub lon_rad: f32,
}

impl RTreeObject for Airport {
    type Envelope = AABB<[f32; 2]>;

    fn envelope(&self) -> Self::Envelope {
        AABB::from_point([self.lat_rad, self.lon_rad])
    }
}

#[derive(Debug, Clone)]
pub struct Disc {
    pub target: Arc<str>,
    pub hostname: String,
    pub lat: f32,
    pub lon: f32,
    pub radius: f32,
}

#[derive(Debug)]
pub struct OutputRecord {
    pub target: Arc<str>,
    pub vp: String,
    pub vp_lat: f32,
    pub vp_lon: f32,
    pub radius: f32,
    pub pop_iata: String,
    pub pop_lat: f32,
    pub pop_lon: f32,
    pub pop_city: String,
    pub pop_cc: String,
}
