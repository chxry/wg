use std::{slice, mem};
use winit::window::{WindowBuilder, Window};
use winit::event_loop::{EventLoop, ControlFlow};
use winit::event::{Event, WindowEvent};
use winit::dpi::PhysicalSize;
use wgpu::util::DeviceExt;
use egui_wgpu_backend::ScreenDescriptor;
use glam::{Vec3, Vec2, Quat, Mat4,Mat3};
use log::LevelFilter;

type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8UnormSrgb;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const SAMPLES: u32 = 4;

struct Node {
  name: String,
  transform: Transform,
  mesh: Mesh,
  tex: Texture<wgpu::BindGroup>,
}

#[tokio::main]
async fn main() -> Result {
  ezlogger::init(LevelFilter::Warn)?;
  let event_loop = EventLoop::new();
  let window = WindowBuilder::new().build(&event_loop)?;
  let renderer = Renderer::new(&window).await?;

  let size = window.inner_size();
  let mut depth_tex = Texture::new(
    &renderer,
    DEPTH_FORMAT,
    wgpu::TextureUsages::RENDER_ATTACHMENT,
    SAMPLES,
    size.width,
    size.height,
  );
  let mut fb = Texture::new(
    &renderer,
    FORMAT,
    wgpu::TextureUsages::RENDER_ATTACHMENT,
    SAMPLES,
    size.width,
    size.height,
  );

  let (doc, buffers, images) = gltf::import("sponza.glb")?;
  let cam = doc
    .nodes()
    .find_map(|n| {
      n.camera().map(|c| {
        let transform = n.transform().decomposed();
        match c.projection() {
          gltf::camera::Projection::Orthographic(o) => panic!(),
          gltf::camera::Projection::Perspective(p) => Camera::new(
            &renderer,
            transform.0.into(),
            Quat::from_array(transform.1),
            p.yfov(),
            [p.znear(), p.zfar().unwrap_or(100.0)],
          ),
        }
      })
    })
    .unwrap();
  let mut nodes: Vec<_> = doc
    .nodes()
    .filter_map(|n| {
      let name = n.name().unwrap_or("?").to_string();
      let transform = n.transform().decomposed();
      let transform = Transform::new(
        &renderer,
        transform.0.into(),
        Quat::from_array(transform.1),
        transform.2.into(),
      );

      n.mesh().map(|mesh| {
        let primitive = mesh.primitives().next().unwrap();
        let reader = primitive.reader(|b| Some(&buffers[b.index()]));

        let verts: Vec<_> = reader
          .read_positions()
          .unwrap()
          .zip(reader.read_tex_coords(0).unwrap().into_f32())
          .zip(reader.read_normals().unwrap())
          .map(|((p, u), n)| Vertex {
            pos: p.into(),
            uv: u.into(),
            normal: n.into(),
          })
          .collect();
        let indices: Vec<_> = reader.read_indices().unwrap().into_u32().collect();
        let mesh = Mesh::new(&renderer, &verts, &indices);

        let image = &images[primitive
          .material()
          .pbr_metallic_roughness()
          .base_color_texture()
          .unwrap()
          .texture()
          .source()
          .index()];
        let tex = Texture::new(
          &renderer,
          wgpu::TextureFormat::Rgba8UnormSrgb,
          wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
          1,
          image.width(),
          image.height(),
        )
        .create_bind_group(&renderer);
        tex.write(&renderer, &image.to_rgba8());
        Node {
          name,
          transform,
          mesh,
          tex,
        }
      })
    })
    .collect();
  let light = Light::new(&renderer, Vec3::new(0.0, 3.0, 3.0), Vec3::ONE);

  let ctx = egui::Context::default();
  ctx.set_pixels_per_point(window.scale_factor() as _);
  let mut egui_platform = egui_winit::State::new(&event_loop);
  let mut egui_renderer = egui_wgpu_backend::RenderPass::new(&renderer.device, FORMAT, 1);

  event_loop.run(move |event, _, control_flow| match event {
    Event::WindowEvent { event, .. } => {
      if !egui_platform.on_event(&ctx, &event).consumed {
        match event {
          WindowEvent::Resized(size) => {
            renderer.resize(size);
            depth_tex = Texture::new(
              &renderer,
              DEPTH_FORMAT,
              wgpu::TextureUsages::RENDER_ATTACHMENT,
              SAMPLES,
              size.width,
              size.height,
            );
            fb = Texture::new(
              &renderer,
              FORMAT,
              wgpu::TextureUsages::RENDER_ATTACHMENT,
              SAMPLES,
              size.width,
              size.height,
            );
          }
          WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
          _ => {}
        }
      }
    }
    Event::RedrawRequested(..) => {
      ctx.begin_frame(egui_platform.take_egui_input(&window));
      egui::Window::new("hello floppa").show(&ctx, |ui| {
        for node in &nodes {
          ui.collapsing(node.name.clone(), |ui| ui.label("floppa"));
        }
      });
      let egui_out = ctx.end_frame();
      egui_platform.handle_platform_output(&window, &ctx, egui_out.platform_output);
      egui_renderer
        .add_textures(&renderer.device, &renderer.queue, &egui_out.textures_delta)
        .unwrap();
      let size = window.inner_size();
      let size = ScreenDescriptor {
        physical_width: size.width,
        physical_height: size.height,
        scale_factor: window.scale_factor() as _,
      };
      let egui_shapes = ctx.tessellate(egui_out.shapes);
      egui_renderer.update_buffers(&renderer.device, &renderer.queue, &egui_shapes, &size);

      let surface = renderer.surface.get_current_texture().unwrap();
      let surface_view = surface
        .texture
        .create_view(&wgpu::TextureViewDescriptor::default());
      let mut encoder = renderer
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

      let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
          view: &fb.view,
          resolve_target: Some(&surface_view),
          ops: wgpu::Operations {
            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
            store: true,
          },
        })],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
          view: &depth_tex.view,
          depth_ops: Some(wgpu::Operations {
            load: wgpu::LoadOp::Clear(1.0),
            store: true,
          }),
          stencil_ops: None,
        }),
        label: None,
      });
      render_pass.set_pipeline(&renderer.pipeline);
      cam.bind(
        &renderer,
        &mut render_pass,
        surface.texture.width() as f32 / surface.texture.height() as f32,
      );
      light.bind(&renderer, &mut render_pass);
      for node in &nodes {
        node.tex.bind(&mut render_pass);
        node.transform.bind(&renderer, &mut render_pass);
        node.mesh.render(&mut render_pass);
      }
      drop(render_pass);

      egui_renderer
        .execute(&mut encoder, &surface_view, &egui_shapes, &size, None)
        .unwrap();

      renderer.queue.submit([encoder.finish()]);
      surface.present();
    }
    Event::MainEventsCleared => window.request_redraw(),
    _ => {}
  });
}

struct Renderer {
  surface: wgpu::Surface,
  device: wgpu::Device,
  queue: wgpu::Queue,
  sampler: wgpu::Sampler,
  pipeline: wgpu::RenderPipeline,
}

impl Renderer {
  async fn new(window: &Window) -> Result<Self> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
    let surface = unsafe { instance.create_surface(&window)? };
    let adapter = instance
      .request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: Some(&surface),
        force_fallback_adapter: false,
      })
      .await
      .unwrap();
    let (device, queue) = adapter
      .request_device(
        &wgpu::DeviceDescriptor {
          features: wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
          limits: wgpu::Limits::default(),
          label: None,
        },
        None,
      )
      .await?;

    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
      address_mode_u: wgpu::AddressMode::ClampToEdge,
      address_mode_v: wgpu::AddressMode::ClampToEdge,
      address_mode_w: wgpu::AddressMode::ClampToEdge,
      mag_filter: wgpu::FilterMode::Linear,
      min_filter: wgpu::FilterMode::Nearest,
      mipmap_filter: wgpu::FilterMode::Nearest,
      ..Default::default()
    });

    let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
      layout: None,
      vertex: wgpu::VertexState {
        module: &shader,
        entry_point: "vs_main",
        buffers: &[Vertex::DESC],
      },
      fragment: Some(wgpu::FragmentState {
        module: &shader,
        entry_point: "fs_main",
        targets: &[Some(wgpu::ColorTargetState {
          format: FORMAT,
          blend: Some(wgpu::BlendState::REPLACE),
          write_mask: wgpu::ColorWrites::ALL,
        })],
      }),
      primitive: wgpu::PrimitiveState::default(),
      depth_stencil: Some(wgpu::DepthStencilState {
        format: DEPTH_FORMAT,
        depth_write_enabled: true,
        depth_compare: wgpu::CompareFunction::Less,
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState::default(),
      }),
      multisample: wgpu::MultisampleState {
        count: SAMPLES,
        mask: !0,
        alpha_to_coverage_enabled: false,
      },
      multiview: None,
      label: None,
    });

    let r = Self {
      surface,
      device,
      queue,
      sampler,
      pipeline,
    };
    r.resize(window.inner_size());
    Ok(r)
  }

  fn resize(&self, size: PhysicalSize<u32>) {
    self.surface.configure(
      &self.device,
      &wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: FORMAT,
        width: size.width,
        height: size.height,
        present_mode: wgpu::PresentMode::Fifo,
        alpha_mode: wgpu::CompositeAlphaMode::Auto,
        view_formats: vec![],
      },
    );
  }
}

#[repr(C)]
struct Vertex {
  pos: Vec3,
  uv: Vec2,
  normal: Vec3,
}

impl Vertex {
  const DESC: wgpu::VertexBufferLayout<'_> = wgpu::VertexBufferLayout {
    array_stride: mem::size_of::<Self>() as _,
    step_mode: wgpu::VertexStepMode::Vertex,
    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x2, 2 => Float32x3],
  };
}
struct Mesh {
  vert_buf: wgpu::Buffer,
  idx_buf: wgpu::Buffer,
  len: u32,
}

impl Mesh {
  fn new(renderer: &Renderer, verts: &[Vertex], indices: &[u32]) -> Self {
    Self {
      vert_buf: renderer
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
          contents: unsafe { cast_slice(verts) },
          usage: wgpu::BufferUsages::VERTEX,
          label: None,
        }),
      idx_buf: renderer
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
          contents: unsafe { cast_slice(indices) },
          usage: wgpu::BufferUsages::INDEX,
          label: None,
        }),
      len: indices.len() as _,
    }
  }

  fn render<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
    render_pass.set_vertex_buffer(0, self.vert_buf.slice(..));
    render_pass.set_index_buffer(self.idx_buf.slice(..), wgpu::IndexFormat::Uint32);
    render_pass.draw_indexed(0..self.len, 0, 0..1);
  }
}

unsafe fn cast_slice<T>(t: &[T]) -> &[u8] {
  slice::from_raw_parts(t.as_ptr() as _, mem::size_of_val(t))
}

struct Texture<B = ()> {
  texture: wgpu::Texture,
  view: wgpu::TextureView,
  bind_group: B,
}

impl Texture {
  fn new(
    renderer: &Renderer,
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
    sample_count: u32,
    width: u32,
    height: u32,
  ) -> Self {
    let texture = renderer.device.create_texture(&wgpu::TextureDescriptor {
      size: wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
      },
      mip_level_count: 1,
      sample_count,
      dimension: wgpu::TextureDimension::D2,
      format,
      usage,
      view_formats: &[],
      label: None,
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    Self {
      texture,
      view,
      bind_group: (),
    }
  }

  fn create_bind_group(self, renderer: &Renderer) -> Texture<wgpu::BindGroup> {
    let bind_group = renderer
      .device
      .create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &renderer.pipeline.get_bind_group_layout(0),
        entries: &[
          wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(&self.view),
          },
          wgpu::BindGroupEntry {
            binding: 1,
            resource: wgpu::BindingResource::Sampler(&renderer.sampler),
          },
        ],
        label: None,
      });
    Texture {
      texture: self.texture,
      view: self.view,
      bind_group,
    }
  }
}

impl<B> Texture<B> {
  fn write(&self, renderer: &Renderer, data: &[u8]) {
    renderer.queue.write_texture(
      wgpu::ImageCopyTexture {
        texture: &self.texture,
        mip_level: 0,
        origin: wgpu::Origin3d::ZERO,
        aspect: wgpu::TextureAspect::All,
      },
      data,
      wgpu::ImageDataLayout {
        offset: 0,
        bytes_per_row: Some(self.texture.format().block_size(None).unwrap() * self.texture.width()),
        rows_per_image: Some(self.texture.height()),
      },
      self.texture.size(),
    );
  }
}

impl Texture<wgpu::BindGroup> {
  fn bind<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
    render_pass.set_bind_group(0, &self.bind_group, &[]);
  }
}

struct Camera {
  pos: Vec3,
  rot: Quat,
  fov: f32,
  clip: [f32; 2],
  buf: wgpu::Buffer,
  bind_group: wgpu::BindGroup,
}

impl Camera {
  fn new(renderer: &Renderer, pos: Vec3, rot: Quat, fov: f32, clip: [f32; 2]) -> Self {
    let buf = renderer.device.create_buffer(&wgpu::BufferDescriptor {
      size: mem::size_of::<CameraUniform>() as _,
      usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
      mapped_at_creation: false,
      label: None,
    });
    let bind_group = renderer
      .device
      .create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &renderer.pipeline.get_bind_group_layout(2),
        entries: &[wgpu::BindGroupEntry {
          binding: 0,
          resource: buf.as_entire_binding(),
        }],
        label: None,
      });
    Self {
      pos,
      rot,
      fov,
      clip,
      buf,
      bind_group,
    }
  }

  fn bind<'a>(&'a self, renderer: &Renderer, render_pass: &mut wgpu::RenderPass<'a>, aspect: f32) {
    renderer.queue.write_buffer(&self.buf, 0, unsafe {
      cast_slice(&[CameraUniform {
        view: Mat4::look_to_rh(self.pos, self.rot * Vec3::NEG_Z, Vec3::Z),
        proj: Mat4::perspective_rh(self.fov, aspect, self.clip[0], self.clip[1]),
      }])
    });
    render_pass.set_bind_group(2, &self.bind_group, &[]);
  }
}

#[derive(Debug)]
#[repr(C)]
struct CameraUniform {
  view: Mat4,
  proj: Mat4,
}

struct Transform {
  pos: Vec3,
  rot: Quat,
  scale: Vec3,
  buf: wgpu::Buffer,
  bind_group: wgpu::BindGroup,
}

impl Transform {
  fn new(renderer: &Renderer, pos: Vec3, rot: Quat, scale: Vec3) -> Self {
    let buf = renderer.device.create_buffer(&wgpu::BufferDescriptor {
      size: mem::size_of::<TransformUniform>() as _,
      usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
      mapped_at_creation: false,
      label: None,
    });
    let bind_group = renderer
      .device
      .create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &renderer.pipeline.get_bind_group_layout(1),

        entries: &[wgpu::BindGroupEntry {
          binding: 0,
          resource: buf.as_entire_binding(),
        }],
        label: None,
      });
    Self {
      pos,
      rot,
      scale,
      buf,
      bind_group,
    }
  }

  fn bind<'a>(&'a self, renderer: &Renderer, render_pass: &mut wgpu::RenderPass<'a>) {
    renderer.queue.write_buffer(&self.buf, 0, unsafe {
      cast_slice(&[TransformUniform {
        model: Mat4::from_scale_rotation_translation(self.scale, self.rot, self.pos),
        normal: Mat3::from_quat(self.rot)
      }])
    });
    render_pass.set_bind_group(1, &self.bind_group, &[]);
  }
}

#[repr(C)]
struct TransformUniform {
  model: Mat4,
  normal: Mat3
}

struct Light {
  pos: Vec3,
  color: Vec3,
  buf: wgpu::Buffer,
  bind_group: wgpu::BindGroup,
}

impl Light {
  fn new(renderer: &Renderer, pos: Vec3, color: Vec3) -> Self {
    let buf = renderer.device.create_buffer(&wgpu::BufferDescriptor {
      size: mem::size_of::<LightUniform>() as _,
      usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
      mapped_at_creation: false,
      label: None,
    });
    let bind_group = renderer
      .device
      .create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &renderer.pipeline.get_bind_group_layout(3),
        entries: &[wgpu::BindGroupEntry {
          binding: 0,
          resource: buf.as_entire_binding(),
        }],
        label: None,
      });
    Self {
      pos,
      color,
      buf,
      bind_group,
    }
  }

  fn bind<'a>(&'a self, renderer: &Renderer, render_pass: &mut wgpu::RenderPass<'a>) {
    renderer.queue.write_buffer(&self.buf, 0, unsafe {
      cast_slice(&[LightUniform {
        pos: self.pos,
        __: 0,
        color: self.color,
        ___: 0,
      }])
    });
    render_pass.set_bind_group(3, &self.bind_group, &[]);
  }
}

#[repr(C)]
struct LightUniform {
  pos: Vec3,
  __: u32,
  color: Vec3,
  ___: u32,
}
