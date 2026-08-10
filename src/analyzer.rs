use rstar::{AABB, RTree};
use std::cmp::Ordering;
use std::collections::HashSet;

use crate::geo::{EARTH_RADIUS_KM, haversine_batch, haversine_distance};
use crate::model::{Airport, Disc, OutputRecord};

/// Maximum number of candidates for which we compute the `candidate_diameter`
const EXACT_DIAMETER_MAX: usize = 512;

/// Upper limit for the number of computations done for `candidate_diameter`
const DIAMETER_SWEEPS: usize = 8;

/// Result of geolocating a single MIS cluster.
pub struct GeolocationResult {
    pub airport: Airport,
    /// Max pairwise distance (km) between surviving candidate cities
    pub candidate_diameter: f32,
    /// Number of discs that successfully narrowed the candidate set
    pub num_constraints: u32,
}

/// Largest great-circle distance (km) between any two of the given locations.
///
/// Computes all pairwise distances for small candidate sets.
/// For large candidate sets (> `EXACT_DIAMETER_MAX`) we use a farthest-point sweep
/// to maintain fast geolocation as this is computation is exponential in cost.
///
/// Farthest-point sweep is implemented by computing all distances from a single point
/// to find its farthest point. This is iteratively repeated using that farthest point
/// till it fails to beat the farthest distance seen.
fn max_pairwise_distance(lats: &[f32], lons: &[f32]) -> f32 {
    debug_assert_eq!(lats.len(), lons.len());
    let n = lats.len();
    // Only a single location in the candidate set.
    if n < 2 {
        return 0.0;
    }

    // Small sets: compare every pair
    if n <= EXACT_DIAMETER_MAX {
        let mut max_dist: f32 = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                let d = haversine_distance(lats[i], lons[i], lats[j], lons[j]);
                if d > max_dist {
                    max_dist = d;
                }
            }
        }
        return max_dist;
    }

    // Large sets: walk to the farthest location until no sweep improves on the last
    let mut dists = vec![0.0f32; n];
    let mut current = 0usize;
    let mut max_dist: f32 = 0.0;

    // Try up to `DIAMETER_SWEEPS` times to find the farthest points
    for _ in 0..DIAMETER_SWEEPS {
        // Compute all distances for the current location
        haversine_batch(lats[current], lons[current], lats, lons, &mut dists);

        // Get the location farthest from the current location and their pairwise distance
        let (farthest, dist) =
            dists
                .iter()
                .enumerate()
                .fold((current, 0.0f32), |(best_i, best_d), (i, &d)| {
                    if d > best_d { (i, d) } else { (best_i, best_d) }
                });

        // If we fail to find a more distant distance we finish
        if dist <= max_dist {
            break;
        }

        // Repeat the algorithm using the farthest location found
        max_dist = dist;
        current = farthest;
    }

    max_dist
}

pub struct AnycastAnalyzer<'a> {
    /// Alpha parameter (distance/population tuner)
    alpha: f32,
    /// Optional pop ratio filter (filter out relatively small locations)
    pop_ratio: f32,
    /// RTree mapping of possible locations
    airport_tree: &'a RTree<Airport>,
    /// Group of measurements belonging to a single target
    all_discs: Vec<Disc>,
    /// Boolean, returns only geolocation for anycast targets if true
    anycast_only: bool,
    /// Whether to compute accuracy metrics
    accuracy: bool,
}

impl<'a> AnycastAnalyzer<'a> {
    /// Init values
    /// Sorts discs from lowest to highest
    pub fn new(
        mut discs: Vec<Disc>,
        airport_tree: &'a RTree<Airport>,
        alpha: f32,
        pop_ratio: f32,
        anycast_only: bool,
        accuracy: bool,
    ) -> Self {
        // Sort discs from lowest to highest
        discs.sort_unstable_by(|a, b| a.radius.partial_cmp(&b.radius).unwrap());
        Self {
            alpha,
            pop_ratio,
            airport_tree,
            all_discs: discs,
            anycast_only,
            accuracy,
        }
    }

    /// Perform the geolocation algorithm.
    /// Returns the set of PoPs found (one if unicast)
    pub fn analyze(self) -> Vec<OutputRecord> {
        let target_ip = self.all_discs.first().unwrap().target.clone();
        // Get MIS discs (each MIS reaches a unique anycast PoP)
        let (num_sites, mis_indices) = self.enumeration();

        // Return early if no PoP is found, or it is unicast and we only care about anycast
        if num_sites == 0 || (self.anycast_only && num_sites <= 1) {
            return vec![];
        }
        // Calculate the distances of all discs to all MIS discs
        let distances = self.precompute_mis_distances(&mis_indices);

        // Init results
        let mut results = Vec::new();
        let mut chosen_airports = HashSet::new();

        // Geolocate each MIS disc
        for (mis_j, disc_index) in mis_indices.iter().enumerate() {
            // Get this MIS disc
            let disc_in_mis = &self.all_discs[*disc_index];

            // Get all discs that touch this MIS, and only this MIS
            let cluster = self.build_cluster_fast(mis_j, &mis_indices, &distances);
            // Geolocate for this MIS cluster
            let geolocation_result = self.geolocation(&cluster);

            // Return the geolocation result (in expected output format)
            if let Some(geo) = geolocation_result {
                if chosen_airports.contains(&geo.airport.iata) {
                    continue;
                }
                chosen_airports.insert(geo.airport.iata.clone());

                results.push(OutputRecord {
                    target: target_ip.clone(),
                    vp: disc_in_mis.hostname.clone(),
                    vp_lat: disc_in_mis.lat.to_degrees(),
                    vp_lon: disc_in_mis.lon.to_degrees(),
                    radius: disc_in_mis.radius,
                    pop_iata: geo.airport.iata.clone(),
                    pop_lat: geo.airport.lat,
                    pop_lon: geo.airport.lon,
                    pop_city: geo.airport.city.clone(),
                    pop_cc: geo.airport.country_code.clone(),
                    candidate_diameter: self.accuracy.then_some(geo.candidate_diameter),
                    num_constraints: self.accuracy.then_some(geo.num_constraints),
                });
            } else {
                results.push(OutputRecord {
                    target: target_ip.clone(),
                    vp: disc_in_mis.hostname.clone(),
                    vp_lat: disc_in_mis.lat.to_degrees(),
                    vp_lon: disc_in_mis.lon.to_degrees(),
                    radius: disc_in_mis.radius,
                    pop_iata: "NoCity".to_string(),
                    pop_lat: disc_in_mis.lat.to_degrees(),
                    pop_lon: disc_in_mis.lon.to_degrees(),
                    pop_city: "N/A".to_string(),
                    pop_cc: "N/A".to_string(),
                    candidate_diameter: self.accuracy.then_some(0.0),
                    num_constraints: self.accuracy.then_some(0),
                });
            }
        }

        results
    }

    /// Get all MIS discs (these all must reach its own anycast PoP)
    /// Achieved using a greedy algorithm (see iGreedy paper)
    /// Return the number of MIS discs (i.e., number of PoPs)
    /// And return the indices of the MIS discs themselves
    fn enumeration(&self) -> (usize, Vec<usize>) {
        // Get the coordinates and radii of all discs
        let mut mis_indices: Vec<usize> = Vec::new();
        let mut mis_lats: Vec<f32> = Vec::new();
        let mut mis_lons: Vec<f32> = Vec::new();
        let mut mis_radii: Vec<f32> = Vec::new();
        let mut batch_dists: Vec<f32> = Vec::new();

        // Starting with the lowest RTT discs, see which discs are its own MIS
        for (i, candidate) in self.all_discs.iter().enumerate() {
            let m = mis_indices.len();
            if m > 0 {
                // If m == 0: first disc (lowest RTT) is always an MIS disc
                batch_dists.resize(m, 0.0);
                // Calculate distance to each MIS disc found so far
                haversine_batch(
                    candidate.lat,
                    candidate.lon,
                    &mis_lats[..m],
                    &mis_lons[..m],
                    &mut batch_dists[..m],
                );
                // Find out whether this disc is overlapping with any MIS disc
                let is_overlapping =
                    (0..m).any(|j| batch_dists[j] <= candidate.radius + mis_radii[j]);
                if is_overlapping {
                    // If this disc is overlapping, it is not an MIS
                    continue;
                }
            }

            // Store this disc as MIS disc
            mis_indices.push(i);
            mis_lats.push(candidate.lat);
            mis_lons.push(candidate.lon);
            mis_radii.push(candidate.radius);
        }
        (mis_indices.len(), mis_indices)
    }

    /// Precompute a distance matrix.
    /// From every disc, to every MIS disc
    fn precompute_mis_distances(&self, mis_indices: &[usize]) -> Vec<Vec<f32>> {
        // Get the coordinates of MIS discs
        let mis_lats: Vec<f32> = mis_indices.iter().map(|&i| self.all_discs[i].lat).collect();
        let mis_lons: Vec<f32> = mis_indices.iter().map(|&i| self.all_discs[i].lon).collect();
        let m = mis_indices.len();

        // For each disc, compute the haversine distance to each MIS disc
        self.all_discs
            .iter()
            .map(|d| {
                let mut dists = vec![0.0f32; m];
                haversine_batch(d.lat, d.lon, &mis_lats, &mis_lons, &mut dists);
                dists
            })
            .collect()
    }

    /// Build a cluster for a particular MIS disc
    /// A cluster is a set of discs that overlap with this MIS disc
    /// We only consider discs that do not overlap other MIS
    fn build_cluster_fast<'s>(
        &'s self,
        mis_j: usize,
        mis_indices: &[usize],
        distances: &[Vec<f32>],
    ) -> Vec<&'s Disc> {
        // Iterate over all discs
        self.all_discs
            .iter()
            .enumerate()
            .filter(|(i, d)| {
                // Get the MIS disc
                let mis_disc = &self.all_discs[mis_indices[mis_j]];
                // Check whether the current disc overlaps the MIS disc
                let overlaps_target = distances[*i][mis_j] <= mis_disc.radius + d.radius;
                if !overlaps_target {
                    // No overlap means it is not in the cluster
                    return false;
                }
                // Count the number of MIS discs that this current disc touches
                let overlap_count = mis_indices
                    .iter()
                    .enumerate()
                    .filter(|(j, mis_idx)| {
                        let md = &self.all_discs[**mis_idx];
                        distances[*i][*j] <= md.radius + d.radius
                    })
                    .count();
                // Return only discs that only touch this MIS (count == 1)
                overlap_count == 1
            })
            .map(|(_, d)| d)
            .collect()
    }

    /// Geolocate the best location for an MIS cluster of discs
    fn geolocation(&self, cluster: &[&Disc]) -> Option<GeolocationResult> {
        // Get the MIS for this cluster (smallest circle)
        let smallest = cluster
            .iter()
            .min_by(|a, b| a.radius.partial_cmp(&b.radius).unwrap())?;

        // Get the coords of the MIS circle
        let center_lat = smallest.lat;
        let center_lon = smallest.lon;

        // Create a bounding box (square) based on its RTT to the target
        let delta_lat = smallest.radius / EARTH_RADIUS_KM;
        let min_lat = center_lat - delta_lat;
        let max_lat = center_lat + delta_lat;

        let delta_lon = smallest.radius / (EARTH_RADIUS_KM * center_lat.cos());
        let min_lon = center_lon - delta_lon;
        let max_lon = center_lon + delta_lon;

        let bbox = AABB::from_corners([min_lat, min_lon], [max_lat, max_lon]);

        // Get all eligible locations within the box (alongside their distance to the center)
        let airports_in_bbox: Vec<(&Airport, f32)> = self
            .airport_tree
            .locate_in_envelope(bbox)
            .map(|a| {
                let dist = haversine_distance(center_lat, center_lon, a.lat_rad, a.lon_rad);
                (a, dist)
            })
            .filter(|(_, dist)| *dist <= smallest.radius) // filter within circle
            .collect();

        // If no locations found -> return early
        if airports_in_bbox.is_empty() {
            return None;
        }

        // Sort cluster by RTT (smallest discs first)
        let mut sorted_cluster: Vec<&&Disc> = cluster.iter().collect();
        sorted_cluster.sort_unstable_by(|a, b| a.radius.partial_cmp(&b.radius).unwrap());

        // Get eligible locations lats and lons
        let apt_lats: Vec<f32> = airports_in_bbox.iter().map(|(a, _)| a.lat_rad).collect();
        let apt_lons: Vec<f32> = airports_in_bbox.iter().map(|(a, _)| a.lon_rad).collect();
        let n_apts = airports_in_bbox.len();

        // Keep track of locations that are valid
        let mut alive: Vec<bool> = vec![true; n_apts];
        let mut prev_alive = alive.clone();
        let mut batch_dists = vec![0.0f32; n_apts];

        // Progressively intersect (reducing bbox size)
        // Start at 1: the bbox pre-filter already applies the smallest disc's constraint
        let mut num_constraints: u32 = 1;
        for disc in &sorted_cluster {
            prev_alive.copy_from_slice(&alive);
            // Calculate distance between current discs and all eligible locations
            haversine_batch(disc.lat, disc.lon, &apt_lats, &apt_lons, &mut batch_dists);

            // Eliminate cities outside of this disc
            let mut changed = false;
            for i in 0..n_apts {
                if alive[i] && batch_dists[i] > disc.radius {
                    alive[i] = false;
                    changed = true;
                }
            }

            // If all cities are eliminated -> break and keep previous alive
            if !alive.iter().any(|&a| a) {
                alive.copy_from_slice(&prev_alive);
                break;
            }

            // Count discs that actually narrowed the candidate set
            if changed {
                num_constraints += 1;
            }
        }

        // Get all eligible locations that survived
        let mut candidates: Vec<(&Airport, f32)> = airports_in_bbox
            .into_iter()
            .zip(alive.iter())
            .filter(|(_, a)| **a)
            .map(|(c, _)| c)
            .collect();

        if candidates.is_empty() {
            return None;
        }

        // Optional relative population filter
        if self.pop_ratio > 0.0 {
            let max_pop = candidates.iter().map(|(a, _)| a.pop).max().unwrap_or(0);
            let min_pop_threshold = (max_pop as f32 * self.pop_ratio) as u32;
            candidates.retain(|(a, _)| a.pop >= min_pop_threshold);
        }

        if candidates.is_empty() {
            return None;
        }

        let total_pop: f32 = candidates.iter().map(|(a, _)| a.pop as f32).sum();
        let total_dist: f32 = candidates.iter().map(|(_, d)| *d).sum();

        // Compute candidate diameter: max pairwise distance between surviving candidates
        let candidate_diameter = if self.accuracy && candidates.len() > 1 {
            let lats: Vec<f32> = candidates.iter().map(|(a, _)| a.lat_rad).collect();
            let lons: Vec<f32> = candidates.iter().map(|(a, _)| a.lon_rad).collect();
            max_pairwise_distance(&lats, &lons)
        } else {
            0.0
        };

        // Score a candidate: relative population minus relative distance to the disc center
        let score = |a: &Airport, d: f32| {
            let pop_score = if total_pop > 0.0 {
                a.pop as f32 / total_pop
            } else {
                0.0
            };
            let dist_score = if total_dist > 0.0 {
                d / total_dist
            } else {
                0.0
            };
            self.alpha * pop_score - (1.0 - self.alpha) * dist_score
        };

        // Return the city with the highest score
        candidates
            .into_iter()
            .max_by(|(a1, d1), (a2, d2)| {
                score(a1, *d1)
                    .partial_cmp(&score(a2, *d2))
                    .unwrap_or(Ordering::Equal)
                    // Tiebreaker (equal scores), smaller distance preferred
                    .then_with(|| d2.partial_cmp(d1).unwrap_or(Ordering::Equal))
                    // Tiebreaker, lexical
                    .then_with(|| a2.iata.cmp(&a1.iata))
            })
            .map(|(a, _)| GeolocationResult {
                airport: a.clone(),
                candidate_diameter,
                num_constraints,
            })
    }
}
