use std::sync::Arc;
use wgpu::util::DeviceExt;
use wgpu::{Adapter, Device, Instance, PresentMode, Queue, Surface, SurfaceCapabilities};
use winit::dpi::PhysicalSize;
use winit::event::{WindowEvent};
use winit::event_loop::{ActiveEventLoop};
use winit::window::{Window, WindowId};
use pollster::FutureExt;
use crate::camera::{Camera, CameraUniform};
use crate::camera_controller::CameraController;
use crate::instance::{InstanceRaw, MyInstance};
use crate::texture;
use cgmath::{Zero, Rotation, InnerSpace};
use cgmath::prelude::*;
//how to acess from src/vertex
use crate::vertex::*;
pub struct StateApplication<'a> {
    pub state: Option<State<'a>>,
}

impl<'a> StateApplication<'a> {
    pub fn new() -> Self {
        Self { state: None }
    }
}

impl<'a> winit::application::ApplicationHandler for StateApplication<'a> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = event_loop
            .create_window(Window::default_attributes().with_title("Hello!"))
            .unwrap();
        self.state = Some(State::new(window));
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let window = self.state.as_ref().unwrap().window();

        if window.id() == window_id {
            if let Some(state) = self.state.as_mut() {
                state.camera_controller.process_events(&event);
            }

            match event {
                WindowEvent::CloseRequested => {
                    event_loop.exit();
                }
                WindowEvent::Resized(physical_size) => {
                    self.state.as_mut().unwrap().resize(physical_size);
                }
                WindowEvent::RedrawRequested => {
                    self.state.as_mut().unwrap().render().unwrap();
                }
                _ => {}
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        let state = self.state.as_mut().unwrap();

        state.camera_controller.update_camera(&mut state.camera);
        state.camera_uniform.update_view_proj(&state.camera);

        state.queue.write_buffer(
            &state.camera_buffer,
            0,
            bytemuck::cast_slice(&[state.camera_uniform]),
        );

        let window = state.window();
        window.request_redraw();
    }
}

pub struct State<'a> {
    pub surface: Surface<'a>,
    pub device: Device,
    pub queue: Queue,
    pub config: wgpu::SurfaceConfiguration,

    pub size: PhysicalSize<u32>,
    pub window: Arc<Window>,
    pub render_pipeline: wgpu::RenderPipeline,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub num_indices: u32,
    pub diffuse_texture: texture::Texture,
    pub diffuse_bind_group: wgpu::BindGroup,
    pub camera: Camera,
    pub camera_uniform: CameraUniform,
    pub camera_buffer: wgpu::Buffer,
    pub camera_bind_group: wgpu::BindGroup,
    pub camera_controller: CameraController,
    pub Myinstances: Vec<MyInstance>,
    pub instance_buffer: wgpu::Buffer,
}

impl<'a> State<'a> {
    pub fn new(window: Window) -> Self {
        let window_arc = Arc::new(window);
        let size = window_arc.inner_size();
        let instance = Self::create_gpu_instance();
        let surface = instance.create_surface(window_arc.clone()).unwrap();
        let adapter = Self::create_adapter(instance, &surface);
        let (device, queue) = Self::create_device(&adapter);
        let surface_caps = surface.get_capabilities(&adapter);
        let config = Self::create_surface_config(size, surface_caps);
        let render_pipeline = Self::create_render_pipeline(&device, &config);
        let vertex_buffer = Self::create_vertex_buffer(&device);
        let index_buffer = Self::create_index_buffer(&device);
        let num_indices = INDICES.len() as u32;

        let diffuse_texture = texture::Texture::from_bytes(
            &device,
            &queue,
            include_bytes!("../happy-tree.png"),
            "happy-tree.png",
        )
        .unwrap();
        let layout = Self::create_texture_bind_group_layout(&device);
        let diffuse_bind_group =
            Self::create_diffuse_bind_group(&device, &layout, &diffuse_texture);
        let camera = Self::create_camera(&config);
        let camera_controller = Self::create_camera_controller(0.2);
        let camera_uniform = Self::create_camera_uniform(&camera);
        let camera_buffer = Self::create_camera_buffer(&device, &camera_uniform);
        let camera_bind_group_layout = Self::create_camera_bind_group_layout(&device);
        let camera_bind_group =
            Self::create_camera_bind_group(&device, &camera_bind_group_layout, &camera_buffer);

        let instance_data = Self::generate_instances(10, cgmath::Vector3::new(5.0, 0.0, 5.0))
            .iter()
            .map(MyInstance::to_raw)
            .collect::<Vec<_>>();
        let Myinstances = Self::generate_instances(10, cgmath::Vector3::new(5.0, 0.0, 5.0));
        let instance_buffer =
            Self::create_instance_buffer(&device, &instance_data, Some("Instance Buffer"));

        Self {
            Myinstances,
            instance_buffer,
            camera_uniform,
            camera,
            camera_buffer,
            camera_bind_group,
            camera_controller,
            diffuse_texture,
            diffuse_bind_group,
            vertex_buffer,
            index_buffer,
            num_indices,
            render_pipeline,
            surface,
            device,
            queue,
            config,
            size,
            window: window_arc,
        }
    }

    fn create_instance_buffer(
        device: &wgpu::Device,
        instance_data: &Vec<InstanceRaw>,
        label: Option<&str>,
    ) -> wgpu::Buffer {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label,
            contents: bytemuck::cast_slice(&instance_data),
            usage: wgpu::BufferUsages::VERTEX,
        })
    }

    fn generate_instances(
        num_instances_per_row: usize,
        instance_displacement: cgmath::Vector3<f32>,
    ) -> Vec<MyInstance> {
        (0..num_instances_per_row)
            .flat_map(|z| {
                (0..num_instances_per_row).map(move |x| {
                    let position = cgmath::Vector3 {
                        x: x as f32,
                        y: 0.0,
                        z: z as f32,
                    } - instance_displacement;

                    let rotation = if position.is_zero() {
                        cgmath::Quaternion::from_axis_angle(
                            cgmath::Vector3::unit_z(),
                            cgmath::Deg(0.0),
                        )
                    } else {
                        cgmath::Quaternion::from_axis_angle(position.normalize(), cgmath::Deg(45.0))
                    };

                    MyInstance { position, rotation }
                })
            })
            .collect::<Vec<_>>()
    }

    fn create_camera_controller(sensitivity: f32) -> CameraController {
        CameraController::new(sensitivity)
    }

    fn create_camera_uniform(camera: &Camera) -> CameraUniform {
        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(camera);
        camera_uniform
    }

    fn create_camera_buffer(device: &wgpu::Device, camera_uniform: &CameraUniform) -> wgpu::Buffer {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[*camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })
    }

    fn create_camera_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            label: Some("camera_bind_group_layout"),
        })
    }

    fn create_camera_bind_group(
        device: &wgpu::Device,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
        camera_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
            label: Some("camera_bind_group"),
        })
    }

    fn create_camera(config: &wgpu::SurfaceConfiguration) -> Camera {
        Camera {
            eye: (0.0, 1.0, 2.0).into(),
            target: (0.0, 0.0, 0.0).into(),
            up: cgmath::Vector3::unit_y(),
            aspect: config.width as f32 / config.height as f32,
            fovy: 45.0,
            znear: 0.1,
            zfar: 100.0,
        }
    }

    fn create_texture_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
            label: Some("texture_bind_group_layout"),
        })
    }

    fn create_diffuse_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        diffuse_texture: &texture::Texture,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&diffuse_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&diffuse_texture.sampler),
                },
            ],
            label: Some("diffuse_bind_group"),
        })
    }

    fn create_vertex_buffer(device: &wgpu::Device) -> wgpu::Buffer {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(crate::vertex::VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        })
    }
    // help me copilot 
    fn create_index_buffer(device: &wgpu::Device) -> wgpu::Buffer {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(crate::vertex::INDICES),
            usage: wgpu::BufferUsages::INDEX,
        })
    }
  // help me to complete this 
  fn create_render_pipeline(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
    });
    let render_pipeline_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[
                &(Self::create_texture_bind_group_layout(&device)),
                &(&(Self::create_camera_bind_group_layout(device))),
            ],
            push_constant_ranges: &[],
        });
    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Render Pipeline"),
        layout: Some(&render_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[Vertex::desc(), InstanceRaw::desc()],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: config.format,
                blend: Some(wgpu::BlendState {
                    color: wgpu::BlendComponent::REPLACE,
                    alpha: wgpu::BlendComponent::REPLACE,
                }),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back),
            // Setting this to anything other than Fill requires Features::POLYGON_MODE_LINE
            // or Features::POLYGON_MODE_POINT
            polygon_mode: wgpu::PolygonMode::Fill,
            // Requires Features::DEPTH_CLIP_CONTROL
            unclipped_depth: false,
            // Requires Features::CONSERVATIVE_RASTERIZATION
            conservative: false,
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        // If the pipeline will be used with a multiview render pass, this
        // indicates how many array layers the attachments will have.
        multiview: None,
        // Useful for optimizing shader compilation on Android
        cache: None,
    });
    render_pipeline
}

fn create_surface_config(
    size: PhysicalSize<u32>,
    capabilities: SurfaceCapabilities,
) -> wgpu::SurfaceConfiguration {
    let surface_format = capabilities
        .formats
        .iter()
        .find(|f| f.is_srgb())
        .copied()
        .unwrap_or(capabilities.formats[0]);

    wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: surface_format,
        width: size.width,
        height: size.height,
        present_mode: PresentMode::AutoNoVsync,
        alpha_mode: capabilities.alpha_modes[0],
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    }
}

fn create_device(adapter: &Adapter) -> (Device, Queue) {
    adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                required_features: wgpu::Features::empty(), //TODO: use what we need extra
                // WebGL doesn't support all of wgpu's features
                required_limits: wgpu::Limits::default(),

                label: None,
                memory_hints: wgpu::MemoryHints::default(),
            },
            None,
        )
        .block_on()
        .unwrap()
}

fn create_adapter(instance: Instance, surface: &Surface) -> Adapter {
    instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(), // may be give option to user
            // to select preference of gpu
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })
        .block_on()
        .unwrap()
}

fn create_gpu_instance() -> Instance {
    Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY, // secondory for open gl , that can also be given to
        // user or not?
        ..Default::default()
    })
}

pub fn update(&mut self) {
    self.camera_controller.update_camera(&mut self.camera);
    self.camera_uniform.update_view_proj(&self.camera);
    self.queue.write_buffer(
        &self.camera_buffer,
        0,
        bytemuck::cast_slice(&[self.camera_uniform]),
    );
}

pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
    self.size = new_size;

    self.config.width = new_size.width;
    self.config.height = new_size.height;

    self.surface.configure(&self.device, &self.config);

    println!("Resized to {:?} from state!", new_size);
}

pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
    let output = self.surface.get_current_texture().unwrap();
    let view = output
        .texture
        .create_view(&wgpu::TextureViewDescriptor::default());

    let mut encoder = self
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

    {
        let mut _render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 1.0,
                        g: 0.2,
                        b: 0.3,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });
        // this part took me upto 4:15 am to debug, dont forget to update these , good night , rust
        // analyzer is good but it needs a lot of memory more than browsers it is heating up my
        // laptop my cpu uptime is 7+ hours i didnt realise i was working continuosly 7 hours
        // on shit like this and my own language probabaly my language will fix these problem
        // of writing 10000+ lines of code to do simple things, gunn nite :)
        _render_pass.set_pipeline(&self.render_pipeline);
        _render_pass.set_bind_group(0, &self.diffuse_bind_group, &[]);
        _render_pass.set_bind_group(1, &self.camera_bind_group, &[]);
        _render_pass.set_vertex_buffer(0, self.instance_buffer.slice(..));
        _render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
        _render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        //            _render_pass.draw_indexed(0..self.num_indices, 0, 0..1);
        _render_pass.draw_indexed(0..self.num_indices, 0, 0..self.Myinstances.len() as _);
    }

    self.queue.submit(std::iter::once(encoder.finish()));
    output.present();

    Ok(())
}

pub fn window(&self) -> &Window {
    &self.window
}
}