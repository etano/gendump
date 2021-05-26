use nannou::image;
use nannou::prelude::*;
use ndarray::prelude::*;
use ndarray::{Array, Ix2};
use rand::{thread_rng, Rng};

struct Model {
    _window: window::Id,
    n_x: usize,
    n_y: usize,
    a: Array<i8, Ix2>,
    rng: rand::rngs::ThreadRng,
    beta: f32,
    texture: wgpu::Texture,
}

fn model(app: &App) -> Model {
    let _window = app.new_window().view(view).build().unwrap();
    let rng = thread_rng();
    let beta: f32 = (2.).sqrt().ln_1p() / 2.;
    let window = app.main_window();
    let wh = window.rect();
    let n_x: usize = wh.w() as usize;
    let n_y: usize = wh.h() as usize;
    let a = Array::<i8, Ix2>::ones((n_x, n_y).f());
    let texture = wgpu::TextureBuilder::new()
        .size([wh.w() as u32, wh.h() as u32])
        .format(wgpu::TextureFormat::Rgba8Unorm)
        .usage(wgpu::TextureUsage::COPY_DST | wgpu::TextureUsage::SAMPLED)
        .build(window.swap_chain_device());

    Model {
        _window,
        n_x,
        n_y,
        a,
        rng,
        beta,
        texture,
    }
}

fn update(_app: &App, _model: &mut Model, _update: Update) {
    let n_x = _model.n_x;
    let n_y = _model.n_y;
    let beta = _model.beta * (10. * _app.mouse.y / _app.window_rect().wh().y).exp();
    for _ in 0..100000 {
        let i: usize = _model.rng.gen_range(0, n_x);
        let j: usize = _model.rng.gen_range(0, n_y);

        // compute energy of current state and new state
        let old_energy = -_model.a[[i, j]]
            * (_model.a[[(i - 1 + n_x) % n_x, j]]
                + _model.a[[(i + 1) % n_x, j]]
                + _model.a[[i, (j - 1 + n_y) % n_y]]
                + _model.a[[i, (j + 1) % n_y]]);
        let new_energy = -old_energy;

        // flip a coin
        if _model.rng.gen::<f32>() < (beta * (old_energy - new_energy) as f32).exp() {
            _model.a[[i, j]] *= -1;
        }
    }
}

fn view(app: &App, _model: &Model, frame: Frame) {
    frame.clear(WHITE);

    let wh = app.window_rect().wh();
    let image = image::ImageBuffer::from_fn(wh.x as u32, wh.y as u32, |i, j| {
        let val = _model.a[[i as usize, j as usize]];
        let r: u8 = if val == -1 { 0 } else { u8::MAX };
        nannou::image::Rgba([r, r, r, u8::MAX])
    });

    let flat_samples = image.as_flat_samples();
    _model.texture.upload_data(
        app.main_window().swap_chain_device(),
        &mut *frame.command_encoder(),
        &flat_samples.as_slice(),
    );

    let draw = app.draw();
    draw.texture(&_model.texture);
    draw.to_frame(app, &frame).unwrap();
}

fn main() {
    nannou::app(model).update(update).run();
}
