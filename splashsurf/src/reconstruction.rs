use crate::{io, log_error};
use anyhow::{anyhow, Context};
use arguments::{
    ReconstructionRunnerArgs, ReconstructionRunnerPathCollection, ReconstructionRunnerPaths,
};
use bytemuck::allocation::cast_vec;
use log::info;
use rayon::prelude::*;
use splashsurf_lib::mesh::{MeshAttribute, MeshWithData, PointCloud3d};
use splashsurf_lib::nalgebra::{Unit, Vector3};
use splashsurf_lib::profile;
use splashsurf_lib::{density_map, Index, Real};
use std::convert::TryFrom;
use std::path::PathBuf;
use structopt::clap::arg_enum;
use structopt::StructOpt;

// TODO: Detect smallest index type (i.e. check if ok to use i32 as index)

/// Command line arguments for the `reconstruct` subcommand
#[derive(Clone, Debug, StructOpt)]
pub struct ReconstructSubcommandArgs {
    /// Path to the input file where the particle positions are stored (supported formats: VTK, binary f32 XYZ, PLY, BGEO)
    #[structopt(short = "-i", long, parse(from_os_str))]
    input_file: Option<PathBuf>,
    /// Path to a sequence of particle files that should be processed, use `{}` in the filename to indicate a placeholder
    #[structopt(short = "-s", long, parse(from_os_str))]
    input_sequence: Option<PathBuf>,
    /// Filename for writing the reconstructed surface to disk (default: "{original_filename}_surface.vtk")
    #[structopt(short = "-o", parse(from_os_str))]
    output_file: Option<PathBuf>,
    /// Optional base directory for all output files (default: current working directory)
    #[structopt(long, parse(from_os_str))]
    output_dir: Option<PathBuf>,
    /// The particle radius of the input data
    #[structopt(long)]
    particle_radius: f64,
    /// The rest density of the fluid
    #[structopt(long, default_value = "1000.0")]
    rest_density: f64,
    /// The smoothing length radius used for the SPH kernel, the kernel compact support radius will be twice the smoothing length (in multiplies of the particle radius)
    #[structopt(long)]
    smoothing_length: f64,
    /// The cube edge length used for marching cubes in multiplies of the particle radius, corresponds to the cell size of the implicit background grid
    #[structopt(long)]
    cube_size: f64,
    /// The iso-surface threshold for the density, i.e. the normalized value of the reconstructed density level that indicates the fluid surface (in multiplies of the rest density)
    #[structopt(long, default_value = "0.6")]
    surface_threshold: f64,
    /// Whether to enable the use of double precision for all computations
    #[structopt(short = "-d", long, default_value = "off", possible_values = &["on", "off"], case_insensitive = true)]
    double_precision: Switch,
    /// Lower corner of the domain where surface reconstruction should be performed, format: domain-min=x_min;y_min;z_min (requires domain-max to be specified)
    #[structopt(
        long,
        number_of_values = 3,
        value_delimiter = ";",
        requires = "domain-max"
    )]
    domain_min: Option<Vec<f64>>,
    /// Upper corner of the domain where surface reconstruction should be performed, format:domain-max=x_max;y_max;z_max (requires domain-min to be specified)
    #[structopt(
        long,
        number_of_values = 3,
        value_delimiter = ";",
        requires = "domain-min"
    )]
    domain_max: Option<Vec<f64>>,
    /// Whether to enable spatial decomposition using an octree (faster) instead of a global approach
    #[structopt(long, default_value = "on", possible_values = &["on", "off"], case_insensitive = true)]
    octree_decomposition: Switch,
    /// Whether to enable stitching of the disconnected local meshes resulting from the reconstruction when spatial decomposition is enabled (slower, but without stitching meshes will not be closed)
    #[structopt(long, default_value = "on", possible_values = &["on", "off"], case_insensitive = true)]
    octree_stitch_subdomains: Switch,
    /// The maximum number of particles for leaf nodes of the octree, default is to compute it based on the number of threads and particles
    #[structopt(long)]
    octree_max_particles: Option<usize>,
    /// Safety factor applied to the kernel compact support radius when it's used as a margin to collect ghost particles in the leaf nodes when performing the spatial decomposition
    #[structopt(long)]
    octree_ghost_margin_factor: Option<f64>,
    /// Whether to compute particle densities in a global step before domain decomposition (slower)
    #[structopt(long, default_value = "off", possible_values = &["on", "off"], case_insensitive = true)]
    octree_global_density: Switch,
    /// Whether to compute particle densities per subdomain but synchronize densities for ghost-particles (faster, recommended).
    /// Note: if both this and global particle density computation is disabled the ghost particle margin has to be increased to at least 2.0
    /// to compute correct density values for ghost particles.
    #[structopt(long, default_value = "on", possible_values = &["on", "off"], case_insensitive = true)]
    octree_sync_local_density: Switch,
    /// Optional filename for writing the point cloud representation of the intermediate density map to disk
    #[structopt(long, parse(from_os_str))]
    output_dm_points: Option<PathBuf>,
    /// Optional filename for writing the grid representation of the intermediate density map to disk
    #[structopt(long, parse(from_os_str))]
    output_dm_grid: Option<PathBuf>,
    /// Optional filename for writing the octree used to partition the particles to disk
    #[structopt(long, parse(from_os_str))]
    output_octree: Option<PathBuf>,
    /// Flag to enable multi-threading to process multiple input files in parallel
    #[structopt(long = "mt-files", default_value = "off", possible_values = &["on", "off"], case_insensitive = true)]
    parallelize_over_files: Switch,
    /// Flag to enable multi-threading for a single input file by processing chunks of particles in parallel
    #[structopt(long = "mt-particles", default_value = "on", possible_values = &["on", "off"], case_insensitive = true)]
    parallelize_over_particles: Switch,
    /// Set the number of threads for the worker thread pool
    #[structopt(long, short = "-n")]
    num_threads: Option<usize>,
    /// Whether to check the final mesh for problems such as holes (note that when stitching is disabled this will lead to a lot of reported problems)
    #[structopt(long, default_value = "off", possible_values = &["on", "off"], case_insensitive = true)]
    check_mesh: Switch,
    /// Whether to write vertex normals to the output file. Note that currently the normals are only computed using an area weighted average of triangle normals.
    #[structopt(long, default_value = "off", possible_values = &["on", "off"], case_insensitive = true)]
    output_normals: Switch,
}

arg_enum! {
    #[derive(Copy, Clone, Debug)]
    pub enum Switch {
        Off,
        On
    }
}

impl Switch {
    fn into_bool(self) -> bool {
        match self {
            Switch::Off => false,
            Switch::On => true,
        }
    }
}

/// Executes the `reconstruct` subcommand
pub fn reconstruct_subcommand(cmd_args: &ReconstructSubcommandArgs) -> Result<(), anyhow::Error> {
    let paths = ReconstructionRunnerPathCollection::try_from(cmd_args)
        .context("Failed parsing input file path(s) from command line")?
        .collect();
    let args = ReconstructionRunnerArgs::try_from(cmd_args)
        .context("Failed processing parameters from command line")?;

    let result = if cmd_args.parallelize_over_files.into_bool() {
        paths.par_iter().try_for_each(|path| {
            reconstruction_pipeline(path, &args)
                .with_context(|| {
                    format!(
                        "Error while processing input file '{}' from a file sequence",
                        path.input_file.display()
                    )
                })
                .map_err(|err| {
                    // Already log the error in case there are multiple errors
                    log_error(&err);
                    err
                })
        })
    } else {
        paths
            .iter()
            .try_for_each(|path| reconstruction_pipeline(path, &args))
    };

    if result.is_ok() {
        info!("Successfully finished processing all inputs.");
    }

    result
}

/// Conversion and validation of command line arguments
mod arguments {
    use super::ReconstructSubcommandArgs;
    use crate::io;
    use anyhow::{anyhow, Context};
    use log::info;
    use splashsurf_lib::nalgebra::Vector3;
    use splashsurf_lib::{AxisAlignedBoundingBox3d, ParticleDensityComputationStrategy};
    use std::convert::TryFrom;
    use std::fs;
    use std::path::{Path, PathBuf};

    /// All arguments that can be supplied to the surface reconstruction tool converted to useful types
    pub struct ReconstructionRunnerArgs {
        pub params: splashsurf_lib::Parameters<f64>,
        pub use_double_precision: bool,
        pub check_mesh: bool,
        pub io_params: io::FormatParameters,
    }

    // Convert raw command line arguments to more useful types
    impl TryFrom<&ReconstructSubcommandArgs> for ReconstructionRunnerArgs {
        type Error = anyhow::Error;

        fn try_from(args: &ReconstructSubcommandArgs) -> Result<Self, Self::Error> {
            // Convert domain args to aabb
            let domain_aabb = match (&args.domain_min, &args.domain_max) {
                (Some(domain_min), Some(domain_max)) => {
                    // This should already be ensured by StructOpt parsing
                    assert_eq!(domain_min.len(), 3);
                    assert_eq!(domain_max.len(), 3);

                    let aabb = AxisAlignedBoundingBox3d::new(
                        Vector3::from_iterator(domain_min.clone()),
                        Vector3::from_iterator(domain_max.clone()),
                    );

                    if !aabb.is_consistent() {
                        return Err(anyhow!("The user specified domain min/max values are inconsistent! min: {:?} max: {:?}", aabb.min().as_slice(), aabb.max().as_slice()));
                    }

                    if aabb.is_degenerate() {
                        return Err(anyhow!(
                            "The user specified domain is degenerate! min: {:?} max: {:?}",
                            aabb.min().as_slice(),
                            aabb.max().as_slice()
                        ));
                    }

                    Some(aabb)
                }
                _ => None,
            };

            // Scale kernel radius and cube size by particle radius
            let compact_support_radius = args.particle_radius * 2.0 * args.smoothing_length;
            let cube_size = args.particle_radius * args.cube_size;

            let spatial_decomposition = if !args.octree_decomposition.into_bool() {
                None
            } else {
                let subdivision_criterion = if let Some(max_particles) = args.octree_max_particles {
                    splashsurf_lib::SubdivisionCriterion::MaxParticleCount(max_particles)
                } else {
                    splashsurf_lib::SubdivisionCriterion::MaxParticleCountAuto
                };
                let ghost_particle_safety_factor = args.octree_ghost_margin_factor;
                let enable_stitching = args.octree_stitch_subdomains.into_bool();

                let particle_density_computation = if args.octree_global_density.into_bool()
                    && args.octree_sync_local_density.into_bool()
                {
                    return Err(anyhow!("Cannot enable both global and merged local particle density computation at the same time. Switch off at least one."));
                } else {
                    if args.octree_global_density.into_bool() {
                        ParticleDensityComputationStrategy::Global
                    } else if args.octree_sync_local_density.into_bool() {
                        ParticleDensityComputationStrategy::SynchronizeSubdomains
                    } else {
                        ParticleDensityComputationStrategy::IndependentSubdomains
                    }
                };

                Some(splashsurf_lib::SpatialDecompositionParameters {
                    subdivision_criterion,
                    ghost_particle_safety_factor,
                    enable_stitching,
                    particle_density_computation,
                })
            };

            // Assemble all parameters for the surface reconstruction
            let params = splashsurf_lib::Parameters {
                particle_radius: args.particle_radius,
                rest_density: args.rest_density,
                compact_support_radius,
                cube_size,
                iso_surface_threshold: args.surface_threshold,
                domain_aabb,
                enable_multi_threading: args.parallelize_over_particles.into_bool(),
                spatial_decomposition,
            };

            // Optionally initialize thread pool
            if let Some(num_threads) = args.num_threads {
                splashsurf_lib::initialize_thread_pool(num_threads)?;
            }

            Ok(ReconstructionRunnerArgs {
                params,
                use_double_precision: args.double_precision.into_bool(),
                check_mesh: args.check_mesh.into_bool(),
                io_params: io::FormatParameters::default(),
            })
        }
    }

    #[derive(Clone, Debug)]
    pub struct ReconstructionRunnerPathCollection {
        is_sequence: bool,
        input_file: PathBuf,
        output_file: PathBuf,
        output_density_map_points_file: Option<PathBuf>,
        output_density_map_grid_file: Option<PathBuf>,
        output_octree_file: Option<PathBuf>,
        output_normals: bool,
    }

    impl ReconstructionRunnerPathCollection {
        fn try_new<P: Into<PathBuf>>(
            is_sequence: bool,
            input_file: P,
            output_base_path: Option<P>,
            output_file: P,
            output_density_map_points_file: Option<P>,
            output_density_map_grid_file: Option<P>,
            output_octree_file: Option<P>,
            output_normals: bool,
        ) -> Result<Self, anyhow::Error> {
            let input_file = input_file.into();
            let output_base_path = output_base_path.map(|p| p.into());
            let output_file = output_file.into();
            let output_density_map_points_file = output_density_map_points_file.map(|p| p.into());
            let output_density_map_grid_file = output_density_map_grid_file.map(|p| p.into());
            let output_octree_file = output_octree_file.map(|p| p.into());

            if let Some(output_base_path) = output_base_path {
                let output_file = output_base_path.join(output_file);

                // Ensure that output directory exists/create it
                if let Some(output_dir) = output_file.parent() {
                    if !output_dir.exists() {
                        info!("The output directory '{}' of the output file '{}' does not exist. Trying to create it now...", output_dir.display(), output_file.display());
                        fs::create_dir_all(output_dir).with_context(|| {
                            format!(
                                "Unable to create output directory '{}'",
                                output_dir.display()
                            )
                        })?;
                    }
                }

                Ok(Self {
                    is_sequence,
                    input_file,
                    output_file,
                    output_density_map_points_file: output_density_map_points_file
                        .map(|f| output_base_path.join(f)),
                    output_density_map_grid_file: output_density_map_grid_file
                        .map(|f| output_base_path.join(f)),
                    output_octree_file: output_octree_file.map(|f| output_base_path.join(f)),
                    output_normals,
                })
            } else {
                Ok(Self {
                    is_sequence,
                    input_file,
                    output_file,
                    output_density_map_points_file,
                    output_density_map_grid_file,
                    output_octree_file,
                    output_normals,
                })
            }
        }

        /// Returns an input/output file path struct for each input file (basically one task per input file)
        pub(crate) fn collect(&self) -> Vec<ReconstructionRunnerPaths> {
            if self.is_sequence {
                let input_file = &self.input_file;
                let output_file = &self.output_file;

                let input_dir = input_file.parent().unwrap();
                let output_dir = output_file.parent().unwrap();

                let input_filename = input_file.file_name().unwrap().to_string_lossy();
                let output_filename = output_file.file_name().unwrap().to_string_lossy();

                let mut paths = Vec::new();
                let mut i: usize = 1;
                loop {
                    let input_filename_i = input_filename.replace("{}", &i.to_string());
                    let input_file_i = input_dir.join(input_filename_i);

                    if input_file_i.is_file() {
                        let output_filename_i = output_filename.replace("{}", &i.to_string());
                        let output_file_i = output_dir.join(output_filename_i);

                        paths.push(ReconstructionRunnerPaths::new(
                            input_file_i,
                            output_file_i,
                            // Don't write density maps etc. when processing a sequence of files
                            None,
                            None,
                            None,
                            self.output_normals,
                        ));
                    } else {
                        break;
                    }

                    i += 1;
                }

                paths
            } else {
                vec![
                    ReconstructionRunnerPaths::new(
                        self.input_file.clone(),
                        self.output_file.clone(),
                        self.output_density_map_points_file.clone(),
                        self.output_density_map_grid_file.clone(),
                        self.output_octree_file.clone(),
                        self.output_normals,
                    );
                    1
                ]
            }
        }
    }

    // Convert input file command line arguments to internal representation
    impl TryFrom<&ReconstructSubcommandArgs> for ReconstructionRunnerPathCollection {
        type Error = anyhow::Error;

        fn try_from(args: &ReconstructSubcommandArgs) -> Result<Self, Self::Error> {
            let output_suffix = "surface";

            if let Some(input_file) = &args.input_file {
                if input_file.is_file() {
                    // Use the user defined output file name if provided...
                    let output_file = if let Some(output_file) = &args.output_file {
                        output_file.clone()
                    // ...otherwise, generate one based on the input filename
                    } else {
                        let input_stem = input_file.file_stem().unwrap().to_string_lossy();
                        format!("{}_{}.vtk", input_stem, output_suffix).into()
                    };

                    Self::try_new(
                        false,
                        input_file.clone(),
                        args.output_dir.clone(),
                        output_file,
                        args.output_dm_points.clone(),
                        args.output_dm_grid.clone(),
                        args.output_octree.clone(),
                        args.output_normals.into_bool(),
                    )
                } else {
                    return Err(anyhow!(
                        "Input file does not exist: \"{}\"",
                        input_file.display()
                    ));
                }
            } else if let Some(input_pattern) = &args.input_sequence {
                // Make sure that the sequence pattern ends with a filename (and not with a path separator)
                let input_filename = match input_pattern.file_name() {
                    Some(input_filename) => input_filename.to_string_lossy(),
                    None => {
                        return Err(anyhow!(
                            "The input file path '{}' does not end with a filename",
                            input_pattern.display()
                        ))
                    }
                };

                // Make sure that the parent directory of the sequence pattern exists
                if let Some(input_dir) = input_pattern.parent() {
                    if !input_dir.is_dir() && input_dir != Path::new("") {
                        return Err(anyhow!(
                            "The parent directory '{}' of the input file path '{}' does not exist",
                            input_dir.display(),
                            input_pattern.display()
                        ));
                    }
                }

                // Make sure that we have a placeholder '{}' in the filename part of the sequence pattern
                if input_filename.contains("{}") {
                    let input_stem = input_pattern.file_stem().unwrap().to_string_lossy();
                    // Currently, only VTK files are supported for output
                    let output_filename = format!(
                        "{}.vtk",
                        input_stem.replace("{}", &format!("{}_{{}}", output_suffix))
                    );

                    Self::try_new(
                        true,
                        input_pattern.clone(),
                        args.output_dir.clone(),
                        output_filename.into(),
                        args.output_dm_points.clone(),
                        args.output_dm_grid.clone(),
                        args.output_octree.clone(),
                        args.output_normals.into_bool(),
                    )
                } else {
                    return Err(anyhow!(
                        "The input sequence pattern \"{}\" does not contain a place holder \"{{}}\"", input_pattern.display()
                    ));
                }
            } else {
                return Err(anyhow!(
                    "Neither an input file path or input sequence pattern was provided"
                ));
            }
        }
    }

    /// All file paths that are relevant for running a single surface reconstruction task
    #[derive(Clone, Debug)]
    pub(crate) struct ReconstructionRunnerPaths {
        pub input_file: PathBuf,
        pub output_file: PathBuf,
        pub output_density_map_points_file: Option<PathBuf>,
        pub output_density_map_grid_file: Option<PathBuf>,
        pub output_octree_file: Option<PathBuf>,
        pub output_normals: bool,
    }

    impl ReconstructionRunnerPaths {
        fn new(
            input_file: PathBuf,
            output_file: PathBuf,
            output_density_map_points_file: Option<PathBuf>,
            output_density_map_grid_file: Option<PathBuf>,
            output_octree_file: Option<PathBuf>,
            output_normals: bool,
        ) -> Self {
            ReconstructionRunnerPaths {
                input_file,
                output_file,
                output_density_map_points_file,
                output_density_map_grid_file,
                output_octree_file,
                output_normals,
            }
        }
    }
}

/// Calls the reconstruction pipeline for single or double precision depending on the runtime parameters
pub(crate) fn reconstruction_pipeline(
    paths: &ReconstructionRunnerPaths,
    args: &ReconstructionRunnerArgs,
) -> Result<(), anyhow::Error> {
    if args.use_double_precision {
        info!("Using double precision (f64) for surface reconstruction.");
        reconstruction_pipeline_generic::<i64, f64>(
            paths,
            &args.params,
            &args.io_params,
            args.check_mesh,
        )?;
    } else {
        info!("Using single precision (f32) for surface reconstruction.");
        reconstruction_pipeline_generic::<i64, f32>(
            paths,
            &args.params.try_convert().ok_or(anyhow!(
                "Unable to convert surface reconstruction parameters from f64 to f32."
            ))?,
            &args.io_params,
            args.check_mesh,
        )?;
    }

    Ok(())
}

/// Wrapper for the reconstruction pipeline: loads input file, runs reconstructions, stores output files
pub(crate) fn reconstruction_pipeline_generic<I: Index, R: Real>(
    paths: &ReconstructionRunnerPaths,
    params: &splashsurf_lib::Parameters<R>,
    io_params: &io::FormatParameters,
    check_mesh: bool,
) -> Result<(), anyhow::Error> {
    profile!("surface reconstruction cli");

    // Load particle positions
    let particle_positions = io::read_particle_positions(&paths.input_file, &io_params.input)
        .with_context(|| {
            format!(
                "Failed to load particle positions from file '{}'",
                paths.input_file.display()
            )
        })?;

    // Perform the surface reconstruction
    let reconstruction =
        splashsurf_lib::reconstruct_surface::<I, R>(particle_positions.as_slice(), &params)?;

    let grid = reconstruction.grid();
    let mesh = reconstruction.mesh();

    // Add normals to mesh if requested
    let mesh = if paths.output_normals {
        profile!("compute normals");
        info!("Computing normals for {} vertices...", mesh.vertices.len());

        let tri_normals = {
            profile!("mesh.par_vertex_normals");
            let tri_normals = mesh.par_vertex_normals();

            // Convert unit vectors to plain vectors
            cast_vec::<Unit<Vector3<R>>, Vector3<R>>(tri_normals)
        };

        MeshWithData::new(mesh.clone())
            .with_point_data(MeshAttribute::new_real_vector3("normals", tri_normals))
    } else {
        MeshWithData::new(mesh.clone())
    };

    // Store the surface mesh
    {
        profile!("write surface mesh to file");
        info!(
            "Writing surface mesh to \"{}\"...",
            paths.output_file.to_string_lossy()
        );

        let io_params = io::FormatParameters::default();

        io::write_mesh(&mesh, paths.output_file.clone(), &io_params.output)?;

        info!("Done.");
    }

    // Store octree leaf nodes as hex cells
    if let Some(output_octree_file) = &paths.output_octree_file {
        info!("Writing octree to \"{}\"...", output_octree_file.display());
        io::vtk_format::write_vtk(
            reconstruction
                .octree()
                .unwrap()
                .hexmesh(grid, true)
                .to_unstructured_grid(),
            output_octree_file,
            "mesh",
        )
        .with_context(|| {
            format!(
                "Failed to write octree to output file '{}'",
                output_octree_file.display()
            )
        })?;
        info!("Done.");
    }

    // Store point cloud density map
    if let Some(output_density_map_points_file) = &paths.output_density_map_points_file {
        info!("Constructing density map point cloud...");
        let density_map = reconstruction
            .density_map()
            .ok_or_else(|| anyhow::anyhow!("No density map was created during reconstruction"))?;

        let point_cloud: PointCloud3d<R> = {
            let mut points = Vec::with_capacity(density_map.len());
            density_map.for_each(|flat_point_index, _| {
                let point = grid.try_unflatten_point_index(flat_point_index).unwrap();
                points.push(grid.point_coordinates(&point));
            });

            PointCloud3d::new(points)
        };

        info!(
            "Saving density map point cloud to \"{}\"...",
            output_density_map_points_file.to_string_lossy()
        );

        io::vtk_format::write_vtk(
            &point_cloud,
            output_density_map_points_file,
            "density_map_points",
        )?;

        info!("Done.");
    }

    // Store hex-mesh density map
    if let Some(output_density_map_grid_file) = &paths.output_density_map_grid_file {
        info!("Constructing density map hex mesh...");
        let density_map = reconstruction
            .density_map()
            .ok_or_else(|| anyhow::anyhow!("No density map was created during reconstruction"))?;

        let density_mesh =
            density_map::sparse_density_map_to_hex_mesh(&density_map, &grid, R::zero());

        info!(
            "Saving density map hex mesh to \"{}\"...",
            output_density_map_grid_file.to_string_lossy()
        );

        io::vtk_format::write_vtk(
            density_mesh.to_unstructured_grid(),
            output_density_map_grid_file,
            "density_map",
        )?;

        info!("Done.");
    }

    if check_mesh {
        if let Err(err) = splashsurf_lib::marching_cubes::check_mesh_consistency(grid, &mesh.mesh) {
            return Err(anyhow!("{}", err));
        } else {
            info!("Checked mesh for problems (holes, etc.), no problems were found.");
        }
    }

    Ok(())
}
