//! Native reader for scamper [warts](https://www.caida.org/catalog/software/scamper/) files.
//! Largely generated using Opus 5 (Claude).
//!
//! Support added for the LACeS pipeline that makes use of latency measurements collected using Ark.
//! Specifically, this file parses warts outputs for the dealias method (`iffinder`).
//!
//! We read the following objects:
//! * [`Object::List`] carries the monitor name, which identifies the vantage point.
//! * [`Object::Dealias`] carries the measurements: per probe a transmit timestamp, and
//! per reply the responding address and a receive timestamp.
//!
//! Support for ping/traceroute can be added on request.

use anyhow::{Context, Result, bail};
use flate2::read::GzDecoder;
use indicatif::ParallelProgressIterator;
use polars::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
use std::path::{Path, PathBuf};

use crate::io::{finalize_measurements, progress_bar};
use crate::vps::{Vp, VpTable};

/// Every warts object starts with this magic.
const WARTS_MAGIC: u16 = 0x1205;

// Object types we care about. See `scamper_file_warts.c`.
const OBJ_LIST: u16 = 0x0001;
const OBJ_DEALIAS: u16 = 0x0009;

/// `scamper_dealias_t` method id for radargun, the method Ark's iffinder runs.
const METHOD_RADARGUN: u8 = 3;

/// Field id / byte length tables, mirroring the `warts_var_t` arrays in scamper.
/// A negative length marks a self-describing field: [`LEN_ADDR`] a warts address,
/// [`LEN_ICMPEXT`] a length-prefixed ICMP extension blob, [`LEN_STRING`] a NUL-terminated string.
const LEN_ADDR: i32 = -1;
const LEN_ICMPEXT: i32 = -2;
const LEN_STRING: i32 = -3;

/// `dealias_vars`: list_id, cycle_id, start, method, result, probec, userid, errmsg.
const DEALIAS_VARS: &[i32] = &[4, 4, 8, 1, 1, 4, 4, LEN_STRING];
/// `dealias_radargun_vars`: probedefc, rounds, wait_probe, wait_round, wait_timeout, flags.
const RADARGUN_VARS: &[i32] = &[4, 2, 2, 4, 1, 1];
/// `dealias_probedef_vars`: ..., dst (10), src (11), ...
const PROBEDEF_VARS: &[i32] = &[4, 4, 4, 1, 1, 1, 4, 1, 2, LEN_ADDR, LEN_ADDR, 2, 2, 2];
/// `dealias_probe_vars`: def, tx, replyc, ipid, seq.
const PROBE_VARS: &[i32] = &[4, 8, 2, 2, 4];
/// `dealias_reply_vars`: ..., rx (2), ..., icmp_ext (7), ..., src (10), ...
const REPLY_VARS: &[i32] = &[4, 8, 2, 1, 2, 1, LEN_ICMPEXT, 1, 1, LEN_ADDR, 4, 1, 2];
/// `list_vars`: descr, monitor.
const LIST_VARS: &[i32] = &[LEN_STRING, LEN_STRING];

// Field ids used below, named for readability.
const DEALIAS_METHOD: u16 = 4;
const DEALIAS_PROBEC: u16 = 6;
const RADARGUN_PROBEDEFC: u16 = 1;
const PROBE_TX: u16 = 2;
const PROBE_REPLYC: u16 = 3;
const REPLY_RX: u16 = 2;
const REPLY_SRC: u16 = 10;
const LIST_MONITOR: u16 = 2;

/// Cursor over a record body.
struct Reader<'a> {
    buf: &'a [u8],
    off: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Reader { buf, off: 0 }
    }

    fn u8(&mut self) -> Option<u8> {
        let v = *self.buf.get(self.off)?;
        self.off += 1;
        Some(v)
    }

    fn u16(&mut self) -> Option<u16> {
        let s = self.buf.get(self.off..self.off + 2)?;
        self.off += 2;
        Some(u16::from_be_bytes([s[0], s[1]]))
    }

    fn u32(&mut self) -> Option<u32> {
        let s = self.buf.get(self.off..self.off + 4)?;
        self.off += 4;
        Some(u32::from_be_bytes([s[0], s[1], s[2], s[3]]))
    }

    fn skip(&mut self, n: usize) -> Option<()> {
        self.off = self.off.checked_add(n).filter(|o| *o <= self.buf.len())?;
        Some(())
    }

    /// A NUL-terminated string.
    fn string(&mut self) -> Option<String> {
        let start = self.off;
        while *self.buf.get(self.off)? != 0 {
            self.off += 1;
        }
        let s = String::from_utf8_lossy(&self.buf[start..self.off]).into_owned();
        self.off += 1;
        Some(s)
    }

    /// A warts address: either defined here or referencing a previously defined address.
    fn addr(&mut self, table: &mut Vec<IpAddr>) -> Option<IpAddr> {
        let size = self.u8()?;
        if size == 0 {
            // Previously defined address
            let idx = self.u32()? as usize;
            return table.get(idx).copied();
        }
        // New address
        let kind = self.u8()?;
        let raw = self.buf.get(self.off..self.off + size as usize)?;
        self.off += size as usize;
        let addr = match (kind, raw.len()) {
            (1, 4) => IpAddr::V4(Ipv4Addr::new(raw[0], raw[1], raw[2], raw[3])),
            (2, 16) => {
                let mut octets = [0u8; 16];
                octets.copy_from_slice(raw);
                IpAddr::V6(Ipv6Addr::from(octets))
            }
            // Unable to parse, unspecified address
            _ => IpAddr::V4(Ipv4Addr::UNSPECIFIED),
        };
        table.push(addr);
        Some(addr)
    }
}

/// The fields present in one params block, with their values already located.
///
/// Reused across records to keep the parse allocation-free in steady state.
#[derive(Default)]
struct Fields {
    /// (field id, offset of the value within the record body)
    scalars: Vec<(u16, usize)>,
    /// (field id, resolved address)
    addrs: Vec<(u16, IpAddr)>,
}

impl Fields {
    fn clear(&mut self) {
        self.scalars.clear();
        self.addrs.clear();
    }

    fn at(&self, id: u16) -> Option<usize> {
        self.scalars
            .iter()
            .find(|(i, _)| *i == id)
            .map(|(_, off)| *off)
    }

    fn u8(&self, body: &[u8], id: u16) -> Option<u8> {
        Reader {
            buf: body,
            off: self.at(id)?,
        }
        .u8()
    }

    fn u16(&self, body: &[u8], id: u16) -> Option<u16> {
        Reader {
            buf: body,
            off: self.at(id)?,
        }
        .u16()
    }

    fn u32(&self, body: &[u8], id: u16) -> Option<u32> {
        Reader {
            buf: body,
            off: self.at(id)?,
        }
        .u32()
    }

    /// A warts timeval: seconds then microseconds, both `u32`.
    fn timeval(&self, body: &[u8], id: u16) -> Option<(u32, u32)> {
        let mut r = Reader {
            buf: body,
            off: self.at(id)?,
        };
        Some((r.u32()?, r.u32()?))
    }

    fn addr(&self, id: u16) -> Option<IpAddr> {
        self.addrs.iter().find(|(i, _)| *i == id).map(|(_, a)| *a)
    }

    fn string(&self, body: &[u8], id: u16) -> Option<String> {
        Reader {
            buf: body,
            off: self.at(id)?,
        }
        .string()
    }
}

/// Read one params block, recording where each present field's value lives.
///
/// `vars` gives the byte length of field 1, 2, ... in order.
fn read_params(
    r: &mut Reader<'_>,
    vars: &[i32],
    table: &mut Vec<IpAddr>,
    out: &mut Fields,
) -> Option<()> {
    out.clear();

    // An all-zero first byte means no fields are set, and no length follows.
    if *r.buf.get(r.off)? == 0 {
        r.off += 1;
        return Some(());
    }

    let flags_start = r.off;
    while (*r.buf.get(r.off)? & 0x80) != 0 {
        r.off += 1;
    }
    r.off += 1;
    let flags = &r.buf[flags_start..r.off];

    let params_len = r.u16()? as usize;
    let params_start = r.off;
    let params_end = params_start.checked_add(params_len)?;
    if params_end > r.buf.len() {
        return None;
    }

    let mut inner = Reader {
        buf: r.buf,
        off: params_start,
    };
    'outer: for (byte, &flag) in flags.iter().enumerate() {
        for bit in 0..7u16 {
            if flag & (1 << bit) == 0 {
                continue;
            }
            let id = byte as u16 * 7 + bit + 1;
            let Some(&len) = vars.get(id as usize - 1) else {
                break 'outer;
            };
            let at = inner.off;
            match len {
                LEN_ADDR => {
                    let addr = inner.addr(table)?;
                    out.addrs.push((id, addr));
                }
                LEN_ICMPEXT => {
                    let n = inner.u16()? as usize;
                    inner.skip(n)?;
                }
                LEN_STRING => {
                    inner.string()?;
                    out.scalars.push((id, at));
                }
                n => {
                    inner.skip(n as usize)?;
                    out.scalars.push((id, at));
                }
            }
            if inner.off > params_end {
                return None;
            }
        }
    }

    // End at the block's declared length
    r.off = params_end;
    Some(())
}

/// One reply, reduced to what geolocation needs.
pub struct Reply {
    /// The responding target address
    pub src: IpAddr,
    pub rtt_ms: f64,
}

/// Latency record, one for each VP.
pub enum Record {
    /// A list record naming the monitor that produced the file.
    Monitor(String),
    /// Measurement replies.
    Replies(Vec<Reply>),
    /// A record type this reader does not handle.
    Unsupported,
}

/// Milliseconds between a probe's transmit and a reply's receive timestamp.
fn rtt_ms(tx: (u32, u32), rx: (u32, u32)) -> f64 {
    let secs = i64::from(rx.0) - i64::from(tx.0);
    let usecs = i64::from(rx.1) - i64::from(tx.1);
    secs as f64 * 1e3 + usecs as f64 / 1e3
}

/// Parse one framed object body.
pub fn parse_object(otype: u16, body: &[u8]) -> Result<Record> {
    match otype {
        OBJ_LIST => parse_list(body),
        OBJ_DEALIAS => parse_dealias(body),
        _ => Ok(Record::Unsupported),
    }
}

/// A list record: warts id, human id, name, then params carrying the monitor name.
fn parse_list(body: &[u8]) -> Result<Record> {
    let parse = || -> Option<String> {
        let mut r = Reader::new(body);
        r.u32()?; // warts-assigned list id
        r.u32()?; // human-assigned list id
        r.string()?; // list name
        let mut table = Vec::new();
        let mut fields = Fields::default();
        read_params(&mut r, LIST_VARS, &mut table, &mut fields)?;
        fields.string(body, LIST_MONITOR)
    };
    match parse() {
        Some(monitor) if !monitor.is_empty() => Ok(Record::Monitor(monitor)),
        _ => Ok(Record::Unsupported),
    }
}

/// A dealias record: header params, method-specific params and probedefs, then the probes.
fn parse_dealias(body: &[u8]) -> Result<Record> {
    let mut replies = Vec::new();
    let mut parse = || -> Option<()> {
        let mut r = Reader::new(body);
        // Addresses are scoped to this record, and references index into definition order.
        let mut table: Vec<IpAddr> = Vec::new();
        let mut fields = Fields::default();

        read_params(&mut r, DEALIAS_VARS, &mut table, &mut fields)?;
        let method = fields.u8(body, DEALIAS_METHOD).unwrap_or(0);
        let probec = fields.u32(body, DEALIAS_PROBEC).unwrap_or(0);
        if method != METHOD_RADARGUN {
            return Some(());
        }

        read_params(&mut r, RADARGUN_VARS, &mut table, &mut fields)?;
        let probedefc = fields.u32(body, RADARGUN_PROBEDEFC).unwrap_or(0);

        for _ in 0..probedefc {
            read_params(&mut r, PROBEDEF_VARS, &mut table, &mut fields)?;
        }

        for _ in 0..probec {
            read_params(&mut r, PROBE_VARS, &mut table, &mut fields)?;
            let tx = fields.timeval(body, PROBE_TX)?;
            let replyc = fields.u16(body, PROBE_REPLYC).unwrap_or(0);

            for _ in 0..replyc {
                read_params(&mut r, REPLY_VARS, &mut table, &mut fields)?;
                let (Some(src), Some(rx)) =
                    (fields.addr(REPLY_SRC), fields.timeval(body, REPLY_RX))
                else {
                    continue;
                };
                replies.push(Reply {
                    src,
                    rtt_ms: rtt_ms(tx, rx),
                });
            }
        }
        Some(())
    };

    if parse().is_none() {
        bail!("malformed dealias record");
    }
    Ok(Record::Replies(replies))
}

/// Ceiling for accepted object sizes to avoid corrupted data.
const MAX_OBJECT_BYTES: usize = 512 << 20;

/// Streams framed objects out of a reader, reusing one body buffer.
struct Objects<R: Read> {
    inner: R,
    body: Vec<u8>,
}

impl<R: Read> Objects<R> {
    fn new(inner: R) -> Self {
        Objects {
            inner,
            body: Vec::new(),
        }
    }

    /// Read the next object, or `None` at clean end of file.
    fn next(&mut self) -> Result<Option<(u16, &[u8])>> {
        let mut header = [0u8; 8];
        let mut read = 0;
        while read < header.len() {
            match self.inner.read(&mut header[read..])? {
                0 => break,
                n => read += n,
            }
        }
        if read == 0 {
            return Ok(None);
        }
        if read < header.len() {
            bail!("truncated object header");
        }

        let magic = u16::from_be_bytes([header[0], header[1]]);
        if magic != WARTS_MAGIC {
            bail!("bad object magic {magic:#06x}, not a warts file");
        }
        let otype = u16::from_be_bytes([header[2], header[3]]);
        let len = u32::from_be_bytes([header[4], header[5], header[6], header[7]]) as usize;
        if len > MAX_OBJECT_BYTES {
            bail!("object claims {len} bytes, refusing to allocate; file is likely corrupt");
        }

        self.body.clear();
        self.body.resize(len, 0);
        self.inner
            .read_exact(&mut self.body)
            .context("truncated object body")?;
        Ok(Some((otype, &self.body)))
    }
}

fn open(path: &Path) -> Result<Box<dyn Read>> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let buffered = BufReader::with_capacity(1 << 20, file);
    if path.extension().is_some_and(|e| e == "gz") {
        Ok(Box::new(GzDecoder::new(buffered)))
    } else {
        Ok(Box::new(buffered))
    }
}

/// Get the VP name from the warts file (using scamper's file naming convention).
fn name_from_filename(path: &Path) -> String {
    let mut name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    // Optionally gzipped
    if let Some(stripped) = name.strip_suffix(".gz") {
        name = stripped.to_string();
    }
    // File suffix
    if let Some(stripped) = name.strip_suffix(".warts") {
        name = stripped.to_string();
    }
    name
}

/// All measurement data for a single file
struct FileData {
    vp: Vp,
    /// Minimum RTT per responding address
    min_rtt: HashMap<IpAddr, f32>,
}

/// Enum for reasons why a file produced nothing
enum Skip {
    /// Unknown VP (not in the provided VP list)
    UnknownVp(String),
}

/// Resolve a file's VP and, if it is known, collect its replies.
fn load_file(path: &Path, vps: &VpTable, threshold: u32) -> Result<Result<FileData, Skip>> {
    let fallback = name_from_filename(path);
    let mut objects = Objects::new(open(path)?);
    let mut resolved: Option<Vp> = None;
    let mut min_rtt: HashMap<IpAddr, f32> = HashMap::new();
    let mut seen_measurements = false;

    while let Some((otype, body)) = objects.next()? {
        match parse_object(otype, body)
            .with_context(|| format!("failed to parse {}", path.display()))?
        {
            Record::Monitor(monitor) => {
                // Get the VP name from the warts file
                if let Some(vp) = vps.get(&monitor) {
                    resolved = Some(vp.clone());
                }
            }
            Record::Replies(replies) => {
                if !seen_measurements {
                    seen_measurements = true;
                    if resolved.is_none() {
                        // use the VP name from the file (assuming default scamper naming)
                        match vps.get(&fallback) {
                            Some(vp) => resolved = Some(vp.clone()),
                            // Unknown VP, exit early
                            None => return Ok(Err(Skip::UnknownVp(fallback))),
                        }
                    }
                }
                // Collect all latency measurement data
                for reply in replies {
                    // Skip negative RTTs TODO verification may not be needed
                    if !(reply.rtt_ms > 0.0) {
                        continue;
                    }
                    // Optional upper-limit RTT threshold to speed up algorithm.
                    if threshold > 0 && reply.rtt_ms > f64::from(threshold) {
                        continue;
                    }
                    let rtt = reply.rtt_ms as f32;
                    min_rtt
                        .entry(reply.src)
                        .and_modify(|best| {
                            if rtt < *best {
                                *best = rtt;
                            }
                        })
                        .or_insert(rtt);
                }
            }
            Record::Unsupported => {}
        }
    }

    match resolved {
        Some(vp) => Ok(Ok(FileData { vp, min_rtt })),
        None => Ok(Err(Skip::UnknownVp(fallback))),
    }
}

/// Expand the `--warts` arguments into a list of files.
///
/// Accepts plain files, directories (searched one level deep), and glob patterns.
pub fn expand_paths(paths: &[PathBuf]) -> Result<Vec<PathBuf>> {
    // Path is a single warts file
    fn is_warts(path: &Path) -> bool {
        let name = path.file_name().unwrap_or_default().to_string_lossy();
        name.ends_with(".warts") || name.ends_with(".warts.gz")
    }

    let mut files = Vec::new();
    for path in paths {
        // Path is a directory containing warts files
        if path.is_dir() {
            let mut found: Vec<PathBuf> = std::fs::read_dir(path)
                .with_context(|| format!("failed to read directory {}", path.display()))?
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.is_file() && is_warts(p))
                .collect();
            if found.is_empty() {
                bail!("no .warts or .warts.gz files in {}", path.display());
            }
            found.sort();
            files.extend(found);
        } else if path.is_file() {
            // Pointing to path (not containing .warts)
            files.push(path.clone());
        } else {
            // Glob pattern
            let pattern = path.to_string_lossy();
            let mut found: Vec<PathBuf> = glob::glob(&pattern)
                .with_context(|| format!("invalid path or glob pattern: {pattern}"))?
                .filter_map(|e| e.ok())
                .filter(|p| p.is_file())
                .collect();
            if found.is_empty() {
                bail!("no files matched {pattern}");
            }
            found.sort();
            files.extend(found);
        }
    }

    files.sort();
    files.dedup();
    if files.is_empty() {
        bail!("no warts files to read");
    }
    Ok(files)
}

/// Read warts files into the standard measurement DataFrame
/// (`addr`, `hostname`, `lat`, `lon`, `rtt`).
///
/// Files are parsed in parallel and reduced to one row per (responder, VP).
pub fn load_warts_data(paths: &[PathBuf], vps: &VpTable, threshold: u32) -> Result<DataFrame> {
    let files = expand_paths(paths)?;
    println!("Reading {} warts file(s)...", files.len());

    let pb = progress_bar(files.len() as u64)?;
    let outcomes: Vec<Result<Result<FileData, Skip>>> = files
        .par_iter()
        .progress_with(pb)
        .map(|path| load_file(path, vps, threshold))
        .collect();

    let mut loaded = Vec::new();
    let mut unknown_vps: Vec<String> = Vec::new();
    for outcome in outcomes {
        match outcome? {
            Ok(data) => loaded.push(data),
            Err(Skip::UnknownVp(name)) => unknown_vps.push(name),
        }
    }

    if !unknown_vps.is_empty() {
        unknown_vps.sort();
        unknown_vps.dedup();
        let shown: Vec<&str> = unknown_vps.iter().take(5).map(|s| s.as_str()).collect();
        let more = unknown_vps.len().saturating_sub(shown.len());
        println!(
            "Dropped {} file(s) whose vantage point is absent from the VPs file: {}{}.",
            unknown_vps.len(),
            shown.join(", "),
            if more > 0 {
                format!(", and {more} more")
            } else {
                String::new()
            }
        );
    }

    if loaded.is_empty() {
        bail!("No warts measurements matched a vantage point in the VPs file.");
    }

    let total: usize = loaded.iter().map(|f| f.min_rtt.len()).sum();
    let mut addrs: Vec<String> = Vec::with_capacity(total);
    let mut hostnames: Vec<String> = Vec::with_capacity(total);
    let mut lats: Vec<f32> = Vec::with_capacity(total);
    let mut lons: Vec<f32> = Vec::with_capacity(total);
    let mut rtts: Vec<f32> = Vec::with_capacity(total);

    // Sort files for consistent ordering between runs
    for file in &loaded {
        let mut entries: Vec<(&IpAddr, &f32)> = file.min_rtt.iter().collect();
        entries.sort_unstable_by_key(|(addr, _)| **addr);
        for (addr, rtt) in entries {
            addrs.push(addr.to_string());
            hostnames.push(file.vp.hostname.clone());
            lats.push(file.vp.lat);
            lons.push(file.vp.lon);
            rtts.push(*rtt);
        }
    }

    println!(
        "Read {} measurements from {} vantage point(s).",
        addrs.len(),
        loaded.len()
    );

    let df = DataFrame::new(
        addrs.len(),
        vec![
            Series::new("addr".into(), addrs).into(),
            Series::new("hostname".into(), hostnames).into(),
            Series::new("lat".into(), lats).into(),
            Series::new("lon".into(), lons).into(),
            Series::new("rtt".into(), rtts).into(),
        ],
    )?;

    // Limit to a single RTT per (VP, target address) pair in case of multiple files per VP
    finalize_measurements(df, threshold)
}
