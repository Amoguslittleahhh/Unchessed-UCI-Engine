//! Strict scalar-side loader for the canonical `UNARCHV1` tensor container.
//!
//! This module validates package structure and exposes typed tensor sections.
//! The neural forward pass is implemented separately in
//! `aegis_v4_runtime.rs`; this module remains responsible for package parsing
//! and validation only.

use std::path::Path;

pub const MAGIC: &[u8; 8] = b"UNARCHV1";
pub const VERSION: u16 = 1;
pub const HEADER_SIZE: usize = 64;
pub const ENTRY_SIZE: usize = 200;
pub const ALIGNMENT: usize = 64;
pub const DTYPE_BYTES: u8 = 0;
pub const DTYPE_I8: u8 = 1;
pub const DTYPE_I16: u8 = 2;
pub const DTYPE_I32: u8 = 3;
pub const DTYPE_F32: u8 = 4;
pub const FLAG_QUANTIZED: u32 = 1;
pub const FLAG_METADATA: u32 = 2;

#[derive(Clone, Debug)]
pub struct TensorSection<'a> {
    pub name: &'a str,
    pub dtype: u8,
    pub shape: [u32; 8],
    pub ndim: usize,
    pub flags: u32,
    pub scale: f32,
    pub zero_point: i32,
    pub data: &'a [u8],
}

impl TensorSection<'_> {
    pub fn dimensions(&self) -> &[u32] {
        &self.shape[..self.ndim]
    }

    pub fn is_quantized(&self) -> bool {
        self.flags & FLAG_QUANTIZED != 0
    }
}

#[derive(Clone, Debug)]
pub struct TensorPackage<'a> {
    pub model_uuid: [u8; 16],
    pub sections: Vec<TensorSection<'a>>,
}

impl<'a> TensorPackage<'a> {
    pub fn parse(bytes: &'a [u8]) -> Result<Self, String> {
        if bytes.len() < HEADER_SIZE {
            return Err("truncated UNARCHV1 header".into());
        }
        if &bytes[..8] != MAGIC {
            return Err("bad UNARCHV1 magic".into());
        }
        let version = read_u16(bytes, 8)?;
        let header_size = read_u16(bytes, 10)? as usize;
        let section_count = read_u32(bytes, 12)? as usize;
        let table_bytes = read_u64(bytes, 16)? as usize;
        let payload_bytes = read_u64(bytes, 24)? as usize;
        if version != VERSION || header_size != HEADER_SIZE {
            return Err("unsupported UNARCHV1 version/header".into());
        }
        if section_count > 100_000 || table_bytes != section_count.saturating_mul(ENTRY_SIZE) {
            return Err("invalid UNARCHV1 section count/table size".into());
        }
        let expected = HEADER_SIZE
            .checked_add(table_bytes)
            .and_then(|value| value.checked_add(payload_bytes))
            .ok_or("UNARCHV1 size overflow")?;
        if bytes.len() != expected {
            return Err("UNARCHV1 declared/file size mismatch".into());
        }
        if read_u64(bytes, 56)? != 0 {
            return Err("UNARCHV1 reserved header bits are nonzero".into());
        }
        let table = &bytes[HEADER_SIZE..HEADER_SIZE + table_bytes];
        let payload = &bytes[HEADER_SIZE + table_bytes..];
        if crc32(table) != read_u32(bytes, 48)? {
            return Err("UNARCHV1 table CRC mismatch".into());
        }
        if crc32(payload) != read_u32(bytes, 52)? {
            return Err("UNARCHV1 payload CRC mismatch".into());
        }
        let mut model_uuid = [0u8; 16];
        model_uuid.copy_from_slice(&bytes[32..48]);
        let mut sections = Vec::with_capacity(section_count);
        let mut names = std::collections::HashSet::with_capacity(section_count);
        for index in 0..section_count {
            let entry = &table[index * ENTRY_SIZE..(index + 1) * ENTRY_SIZE];
            let name_len = read_u16(entry, 0)? as usize;
            let dtype = entry[2];
            let ndim = entry[3] as usize;
            let flags = read_u32(entry, 4)?;
            if name_len == 0 || name_len > 128 || ndim > 8 || read_u32(entry, 68)? != 0 {
                return Err(format!("invalid UNARCHV1 entry {index}"));
            }
            if dtype > DTYPE_F32 {
                return Err(format!("unsupported dtype in section {index}"));
            }
            let mut shape = [0u32; 8];
            for (axis, dimension) in shape.iter_mut().enumerate() {
                *dimension = read_u32(entry, 8 + axis * 4)?;
            }
            let offset = read_u64(entry, 40)? as usize;
            let length = read_u64(entry, 48)? as usize;
            let scale = f32::from_bits(read_u32(entry, 56)?);
            let zero_point = read_u32(entry, 60)? as i32;
            let section_crc = read_u32(entry, 64)?;
            let name = std::str::from_utf8(&entry[72..72 + name_len])
                .map_err(|_| format!("section {index} name is not UTF-8"))?;
            if !names.insert(name) {
                return Err(format!("duplicate UNARCHV1 section {name:?}"));
            }
            if !offset.is_multiple_of(ALIGNMENT) {
                return Err(format!("unaligned UNARCHV1 section {name:?}"));
            }
            let end = offset
                .checked_add(length)
                .ok_or_else(|| format!("section {name:?} size overflow"))?;
            if end > payload.len() {
                return Err(format!("section {name:?} outside payload"));
            }
            let data = &payload[offset..end];
            if crc32(data) != section_crc {
                return Err(format!("section {name:?} CRC mismatch"));
            }
            validate_tensor_size(name, dtype, &shape[..ndim], length)?;
            if !scale.is_finite() || scale <= 0.0 {
                return Err(format!("section {name:?} has invalid scale"));
            }
            sections.push(TensorSection {
                name,
                dtype,
                shape,
                ndim,
                flags,
                scale,
                zero_point,
                data,
            });
        }
        if !sections
            .iter()
            .any(|section| section.name == "__metadata__" && section.flags & FLAG_METADATA != 0)
        {
            return Err("UNARCHV1 metadata section missing".into());
        }
        Ok(Self {
            model_uuid,
            sections,
        })
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Vec<u8>, String> {
        std::fs::read(path.as_ref())
            .map_err(|error| format!("read {}: {error}", path.as_ref().display()))
    }

    pub fn section(&self, name: &str) -> Option<&TensorSection<'a>> {
        self.sections.iter().find(|section| section.name == name)
    }
}

fn validate_tensor_size(name: &str, dtype: u8, shape: &[u32], length: usize) -> Result<(), String> {
    let elements = shape.iter().try_fold(1usize, |product, &dimension| {
        product.checked_mul(dimension as usize)
    });
    let elements = elements.ok_or_else(|| format!("section {name:?} shape overflow"))?;
    let width = match dtype {
        DTYPE_BYTES | DTYPE_I8 => 1,
        DTYPE_I16 => 2,
        DTYPE_I32 | DTYPE_F32 => 4,
        _ => return Err(format!("section {name:?} has unknown dtype")),
    };
    let expected = elements
        .checked_mul(width)
        .ok_or_else(|| format!("section {name:?} byte-size overflow"))?;
    if expected != length {
        return Err(format!(
            "section {name:?} byte length {length} != shape/dtype {expected}"
        ));
    }
    Ok(())
}

fn read_u16(bytes: &[u8], offset: usize) -> Result<u16, String> {
    let slice = bytes
        .get(offset..offset + 2)
        .ok_or("truncated UNARCHV1 u16")?;
    Ok(u16::from_le_bytes(slice.try_into().unwrap()))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, String> {
    let slice = bytes
        .get(offset..offset + 4)
        .ok_or("truncated UNARCHV1 u32")?;
    Ok(u32::from_le_bytes(slice.try_into().unwrap()))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64, String> {
    let slice = bytes
        .get(offset..offset + 8)
        .ok_or("truncated UNARCHV1 u64")?;
    Ok(u64::from_le_bytes(slice.try_into().unwrap()))
}

pub fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &byte in bytes {
        crc ^= byte as u32;
        for _ in 0..8 {
            crc = (crc >> 1) ^ (0xedb8_8320u32 & 0u32.wrapping_sub(crc & 1));
        }
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Vec<u8> {
        let metadata = br#"{"architecture":"Unarchitectured v1"}"#;
        let weights = [1i8, -2, 3, -4];
        let mut payload = Vec::new();
        payload.extend_from_slice(metadata);
        payload.resize(64, 0);
        payload.extend(weights.iter().map(|value| *value as u8));
        let mut table = Vec::new();
        table.extend(entry(
            "__metadata__",
            DTYPE_BYTES,
            FLAG_METADATA,
            &[metadata.len() as u32],
            0,
            metadata,
            1.0,
        ));
        table.extend(entry(
            "weight",
            DTYPE_I8,
            FLAG_QUANTIZED,
            &[2, 2],
            64,
            &weights.map(|value| value as u8),
            0.25,
        ));
        let mut header = Vec::new();
        header.extend_from_slice(MAGIC);
        header.extend_from_slice(&VERSION.to_le_bytes());
        header.extend_from_slice(&(HEADER_SIZE as u16).to_le_bytes());
        header.extend_from_slice(&2u32.to_le_bytes());
        header.extend_from_slice(&(table.len() as u64).to_le_bytes());
        header.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        header.extend_from_slice(&[7u8; 16]);
        header.extend_from_slice(&crc32(&table).to_le_bytes());
        header.extend_from_slice(&crc32(&payload).to_le_bytes());
        header.extend_from_slice(&0u64.to_le_bytes());
        assert_eq!(header.len(), HEADER_SIZE);
        [header, table, payload].concat()
    }

    fn entry(
        name: &str,
        dtype: u8,
        flags: u32,
        shape: &[u32],
        offset: u64,
        data: &[u8],
        scale: f32,
    ) -> Vec<u8> {
        let mut output = Vec::with_capacity(ENTRY_SIZE);
        output.extend_from_slice(&(name.len() as u16).to_le_bytes());
        output.push(dtype);
        output.push(shape.len() as u8);
        output.extend_from_slice(&flags.to_le_bytes());
        for axis in 0..8 {
            output.extend_from_slice(&shape.get(axis).copied().unwrap_or(0).to_le_bytes());
        }
        output.extend_from_slice(&offset.to_le_bytes());
        output.extend_from_slice(&(data.len() as u64).to_le_bytes());
        output.extend_from_slice(&scale.to_bits().to_le_bytes());
        output.extend_from_slice(&0i32.to_le_bytes());
        output.extend_from_slice(&crc32(data).to_le_bytes());
        output.extend_from_slice(&0u32.to_le_bytes());
        let mut name_field = [0u8; 128];
        name_field[..name.len()].copy_from_slice(name.as_bytes());
        output.extend_from_slice(&name_field);
        assert_eq!(output.len(), ENTRY_SIZE);
        output
    }

    #[test]
    fn parses_strict_fixture() {
        let bytes = fixture();
        let package = TensorPackage::parse(&bytes).unwrap();
        assert_eq!(package.model_uuid, [7u8; 16]);
        let weights = package.section("weight").unwrap();
        assert_eq!(weights.dimensions(), &[2, 2]);
        assert!(weights.is_quantized());
        assert_eq!(weights.data, &[1, 254, 3, 252]);
    }

    #[test]
    fn rejects_payload_and_section_corruption() {
        let mut bytes = fixture();
        *bytes.last_mut().unwrap() ^= 1;
        assert!(TensorPackage::parse(&bytes).is_err());
    }

    #[test]
    fn rejects_bad_magic_and_truncation() {
        let mut bytes = fixture();
        bytes[0] = b'X';
        assert!(TensorPackage::parse(&bytes).is_err());
        assert!(TensorPackage::parse(&bytes[..20]).is_err());
    }
}
