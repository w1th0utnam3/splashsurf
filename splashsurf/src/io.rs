use anyhow::{anyhow, Context};
use log::info;
use splashsurf_lib::nalgebra::Vector3;
use splashsurf_lib::profile;
use splashsurf_lib::Real;
use splashsurf_lib::{
    mesh::{Mesh3d, MeshWithData, TriMesh3d},
    vtkio::model::DataSet,
};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

pub mod bgeo_format;
pub mod json_format;
pub mod obj_format;
pub mod ply_format;
pub mod vtk_format;
pub mod xyz_format;

/// File format parameters for all io
#[derive(Clone, Debug, Default)]
pub struct FormatParameters {
    pub input: InputFormatParameters,
    pub output: OutputFormatParameters,
}

/// File format parameters for input files
#[derive(Clone, Debug)]
pub struct InputFormatParameters {}

impl Default for InputFormatParameters {
    fn default() -> Self {
        Self {}
    }
}

/// File format parameters for output files
#[derive(Clone, Debug)]
pub struct OutputFormatParameters {}

impl Default for OutputFormatParameters {
    fn default() -> Self {
        Self {}
    }
}

/// Loads particles positions from the given file path, automatically detects the file format
pub fn read_particle_positions<R: Real, P: AsRef<Path>>(
    input_file: P,
    _format_params: &InputFormatParameters,
) -> Result<Vec<Vector3<R>>, anyhow::Error> {
    let input_file = input_file.as_ref();
    info!(
        "Reading particle dataset from \"{}\"...",
        input_file.display()
    );

    let particle_positions = if let Some(extension) = input_file.extension() {
        profile!("loading particle positions");

        let extension = extension
            .to_str()
            .ok_or(anyhow!("Invalid extension of input file"))?;

        match extension.to_lowercase().as_str() {
            "vtk" => vtk_format::particles_from_vtk(&input_file)?,
            "xyz" => xyz_format::particles_from_xyz(&input_file)?,
            "ply" => ply_format::particles_from_ply(&input_file)?,
            "bgeo" => bgeo_format::particles_from_bgeo(&input_file)?,
            "json" => json_format::particles_from_json(&input_file)?,
            _ => {
                return Err(anyhow!(
                    "Unsupported file format extension \"{}\" for reading particles",
                    extension
                ));
            }
        }
    } else {
        return Err(anyhow!(
            "Unable to detect file format of particle input file (file name has to end with supported extension)",
        ));
    };

    info!(
        "Successfully read dataset with {} particle positions.",
        particle_positions.len()
    );

    Ok(particle_positions)
}

/// Writes particles positions to the given file path, automatically detects the file format
pub fn write_particle_positions<R: Real, P: AsRef<Path>>(
    particles: &[Vector3<R>],
    output_file: P,
    _format_params: &OutputFormatParameters,
) -> Result<(), anyhow::Error> {
    let output_file = output_file.as_ref();
    info!(
        "Writing {} particles to \"{}\"...",
        particles.len(),
        output_file.display()
    );

    if let Some(extension) = output_file.extension() {
        profile!("writing particle positions");

        let extension = extension
            .to_str()
            .ok_or(anyhow!("Invalid extension of output file"))?;

        match extension.to_lowercase().as_str() {
            "vtk" => vtk_format::particles_to_vtk(particles, &output_file)?,
            _ => {
                return Err(anyhow!(
                    "Unsupported file format extension \"{}\" for writing particles",
                    extension
                ));
            }
        }
    } else {
        return Err(anyhow!(
            "Unable to detect file format of particle output file (file name has to end with supported extension)",
        ));
    };

    info!("Successfully wrote particles to file.");
    Ok(())
}

/// Loads a surface mesh from the given file path, automatically detects the file format
pub fn read_surface_mesh<R: Real, P: AsRef<Path>>(
    input_file: P,
    _format_params: &InputFormatParameters,
) -> Result<MeshWithData<R, TriMesh3d<R>>, anyhow::Error> {
    let input_file = input_file.as_ref();
    info!("Reading mesh from \"{}\"...", input_file.display());

    let mesh = if let Some(extension) = input_file.extension() {
        profile!("loading surface mesh");

        let extension = extension
            .to_str()
            .ok_or(anyhow!("Invalid extension of input file"))?;

        match extension.to_lowercase().as_str() {
            "vtk" => vtk_format::surface_mesh_from_vtk(&input_file)?,
            "ply" => ply_format::surface_mesh_from_ply(&input_file)?,
            _ => {
                return Err(anyhow!(
                    "Unsupported file format extension \"{}\" for reading surface meshes",
                    extension
                ));
            }
        }
    } else {
        return Err(anyhow!(
            "Unable to detect file format of mesh input file (file name has to end with supported extension)",
        ));
    };

    info!(
        "Successfully read mesh with {} vertices and {} cells.",
        mesh.mesh.vertices().len(),
        mesh.mesh.cells().len()
    );

    Ok(mesh)
}

/// Writes a mesh to the given file path, automatically detects the file format
pub fn write_mesh<'a, R: Real, MeshT: Mesh3d<R>, P: AsRef<Path>>(
    mesh: &'a MeshWithData<R, MeshT>,
    output_file: P,
    _format_params: &OutputFormatParameters,
) -> Result<(), anyhow::Error>
where
    &'a MeshWithData<R, MeshT>: Into<DataSet>,
{
    let output_file = output_file.as_ref();
    info!(
        "Writing mesh with {} vertices and {} cells to \"{}\"...",
        mesh.mesh.vertices().len(),
        mesh.mesh.cells().len(),
        output_file.display()
    );

    if let Some(extension) = output_file.extension() {
        profile!("writing mesh");

        let extension = extension
            .to_str()
            .ok_or(anyhow!("Invalid extension of output file"))?;

        match extension.to_lowercase().as_str() {
            "vtk" => {
                vtk_format::write_vtk(mesh, &output_file, "mesh").with_context(|| {
                    format!(
                        "Failed to write reconstructed surface to output file '{}'",
                        output_file.to_string_lossy()
                    )
                })?;
            }
            "obj" => {
                obj_format::mesh_to_obj(mesh, &output_file).with_context(|| {
                    format!(
                        "Failed to write reconstructed surface to output file '{}'",
                        output_file.to_string_lossy()
                    )
                })?;
            }
            _ => {
                return Err(anyhow!(
                    "Unsupported file format extension \"{}\" for writing mesh to output file '{}'",
                    extension,
                    output_file.to_string_lossy(),
                ));
            }
        }
    } else {
        return Err(anyhow!(
            "Unable to detect file format of mesh output file (file name has to end with supported extension)",
        ));
    };

    info!("Successfully wrote mesh to file.");
    Ok(())
}

#[allow(dead_code)]
pub fn to_binary_f32<R: Real, P: AsRef<Path>>(file: P, values: &[R]) -> Result<(), anyhow::Error> {
    let file = file.as_ref();
    let file = File::create(file).context("Unable to create binary file")?;
    let mut writer = BufWriter::new(file);

    for v in values {
        let v_f32 = v.to_f32().unwrap();
        writer.write(&v_f32.to_ne_bytes())?;
    }

    Ok(())
}
