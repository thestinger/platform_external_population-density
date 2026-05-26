//! Provides utility functions for GeoTIFF file decompression and verification.

#![allow(clippy::collapsible_if)]

use anyhow::{Result, anyhow};
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use tiff::decoder::{Decoder, Limits};

/// Manages automatic cleanup of a temporary file when dropped from scope.
pub struct CleanupGuard {
    pub path: Option<PathBuf>,
}

impl CleanupGuard {
    /// Creates a new guard for a temporary path.
    pub fn new(path: PathBuf) -> Self {
        Self { path: Some(path) }
    }
}

impl Drop for CleanupGuard {
    fn drop(&mut self) {
        if let Some(ref path) = self.path {
            if path.exists() {
                let _ = std::fs::remove_file(path);
            }
        }
    }
}

const TIFF_COMPRESSION_NONE: u16 = 1;
const TIFF_COMPRESSION_LZW: u16 = 5;

/// Checks if a TIFF file is LZW compressed.
pub fn is_lzw_compressed<P: AsRef<Path>>(path: P) -> Result<bool> {
    let file = File::open(path)?;
    let mut decoder = Decoder::new(BufReader::new(file))?.with_limits(Limits::unlimited());
    let compression = match decoder.get_tag_unsigned::<u16>(tiff::tags::Tag::Compression) {
        Ok(compression_tag) => compression_tag,
        Err(_) => TIFF_COMPRESSION_NONE,
    };
    Ok(compression == TIFF_COMPRESSION_LZW)
}

/// Prepares a GeoTIFF file for decoding.
///
/// If the file is LZW compressed, it automatically converts it to Deflate (ZIP)
/// compression on-the-fly using the system utility 'tiffcp' and returns a
/// cleanup guard to delete the temporary file on scope exit.
pub fn prepare_geotiff<P: AsRef<Path>>(
    input_path: P,
    temp_suffix: &str,
) -> Result<(PathBuf, CleanupGuard)> {
    let path_ref = input_path.as_ref();
    let is_lzw = is_lzw_compressed(path_ref)?;
    if is_lzw {
        println!("Detected LZW compression. Performing on-the-fly conversion using tiffcp...");
        let check_command = std::process::Command::new("tiffcp").arg("-i").output();
        if check_command.is_err() {
            return Err(anyhow!(
                "system utility 'tiffcp' is not installed or not found on your PATH.\n\
                 This tool is required to convert LZW-compressed GeoTIFFs to Deflate compression on-the-fly.\n\
                 Please install the standard 'libtiff-tools' package."
            ));
        }

        let temporary_path = path_ref.with_extension(temp_suffix);
        let guard = CleanupGuard::new(temporary_path.clone());

        let status = std::process::Command::new("tiffcp")
            .arg("-m")
            .arg("0")
            .arg("-c")
            .arg("zip")
            .arg(path_ref)
            .arg(&temporary_path)
            .status()
            .map_err(|error| anyhow!("failed to run tiffcp subprocess: {}", error))?;

        if !status.success() {
            return Err(anyhow!(
                "tiffcp failed with exit code: {:?}",
                status.code().unwrap_or(-1)
            ));
        }
        Ok((temporary_path, guard))
    } else {
        Ok((path_ref.to_path_buf(), CleanupGuard { path: None }))
    }
}
