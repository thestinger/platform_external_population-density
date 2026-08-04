//! Provides utility functions for GeoTIFF file decompression and verification.

use anyhow::{Context, Result, anyhow};
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use tempfile::{Builder, TempDir};
use tiff::decoder::{Decoder, Limits};

const PREPARED_TIFF_FILENAME: &str = "prepared.tif";
const TEMPORARY_DIRECTORY_PREFIX: &str = "population-density-geotiff-";
const TIFF_COMPRESSION_NONE: u16 = 1;
const TIFF_COMPRESSION_LZW: u16 = 5;

/// Owns a TIFF path and any temporary directory that contains it.
pub struct PreparedGeoTiff {
    path: PathBuf,
    _temporary_directory: Option<TempDir>,
}

impl PreparedGeoTiff {
    /// Returns the TIFF path that remains valid while this value is alive.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Creates a non-temporary prepared TIFF.
    fn original(path: PathBuf) -> Self {
        Self {
            path,
            _temporary_directory: None,
        }
    }

    /// Creates a temporary prepared TIFF.
    fn temporary(path: PathBuf, temporary_directory: TempDir) -> Self {
        Self {
            path,
            _temporary_directory: Some(temporary_directory),
        }
    }
}

/// Returns whether a TIFF file uses LZW compression.
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
/// Converts LZW input to Deflate with `tiffcp` in an owned temporary directory.
pub fn prepare_geotiff<P: AsRef<Path>>(input_path: P) -> Result<PreparedGeoTiff> {
    let path_ref = input_path.as_ref();
    if !is_lzw_compressed(path_ref)? {
        return Ok(PreparedGeoTiff::original(path_ref.to_path_buf()));
    }

    println!("Detected LZW compression; converting it to Deflate with tiffcp...");
    prepare_converted_geotiff(path_ref, |input_path, output_path| {
        let status = std::process::Command::new("tiffcp")
            .arg("-m")
            .arg("0")
            .arg("-c")
            .arg("zip")
            .arg(input_path)
            .arg(output_path)
            .status()
            .map_err(|error| {
                anyhow!(
                    "failed to run tiffcp for LZW-compressed GeoTIFF '{}': {}",
                    input_path.display(),
                    error
                )
            })?;

        if !status.success() {
            return Err(anyhow!(
                "tiffcp failed for LZW-compressed GeoTIFF '{}' with status {}",
                input_path.display(),
                status
            ));
        }
        Ok(())
    })
}

/// Converts a TIFF into an owned temporary directory.
fn prepare_converted_geotiff(
    input_path: &Path,
    convert: impl FnOnce(&Path, &Path) -> Result<()>,
) -> Result<PreparedGeoTiff> {
    let temporary_directory = Builder::new()
        .prefix(TEMPORARY_DIRECTORY_PREFIX)
        .tempdir()
        .context("failed to create a temporary GeoTIFF directory")?;
    let output_path = temporary_directory.path().join(PREPARED_TIFF_FILENAME);
    convert(input_path, &output_path)?;
    Ok(PreparedGeoTiff::temporary(output_path, temporary_directory))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;
    use std::sync::{Arc, Barrier};

    const CONCURRENT_CONVERSION_COUNT: usize = 8;

    /// Verifies that concurrent conversions own distinct temporary paths.
    #[test]
    fn concurrent_conversions_use_distinct_owned_paths() {
        let source_directory = tempfile::tempdir().unwrap();
        let source_path = source_directory.path().join("source.tif");
        let source_bytes = b"deterministic test TIFF contents";
        std::fs::write(&source_path, source_bytes).unwrap();

        let barrier = Arc::new(Barrier::new(CONCURRENT_CONVERSION_COUNT));
        let threads = (0..CONCURRENT_CONVERSION_COUNT)
            .map(|_| {
                let barrier = Arc::clone(&barrier);
                let source_path = source_path.clone();
                std::thread::spawn(move || {
                    barrier.wait();
                    prepare_converted_geotiff(&source_path, |input_path, output_path| {
                        std::fs::copy(input_path, output_path)?;
                        Ok(())
                    })
                    .unwrap()
                })
            })
            .collect::<Vec<_>>();
        let prepared_tiffs = threads
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .collect::<Vec<_>>();

        let paths = prepared_tiffs
            .iter()
            .map(|prepared_tiff| prepared_tiff.path().to_path_buf())
            .collect::<Vec<_>>();
        assert_eq!(
            paths.iter().collect::<BTreeSet<_>>().len(),
            CONCURRENT_CONVERSION_COUNT
        );
        for path in &paths {
            assert_eq!(std::fs::read(path).unwrap(), source_bytes);
        }

        drop(prepared_tiffs);
        assert!(paths.iter().all(|path| !path.exists()));
    }

    /// Verifies that conversion errors remove their temporary directory.
    #[test]
    fn conversion_error_cleans_up_temporary_directory() {
        let source_directory = tempfile::tempdir().unwrap();
        let source_path = source_directory.path().join("source.tif");
        std::fs::write(&source_path, b"source").unwrap();
        let temporary_path = std::sync::Mutex::new(None);

        let result = prepare_converted_geotiff(&source_path, |_, output_path| {
            *temporary_path.lock().unwrap() = Some(output_path.to_path_buf());
            Err(anyhow!("intentional conversion failure"))
        });
        assert!(result.is_err());
        assert!(!temporary_path.lock().unwrap().as_ref().unwrap().exists());
    }
}
