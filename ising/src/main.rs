use nannou::image;
use nannou::prelude::*;
use ndarray::prelude::*;
use ndarray::{Array, Ix2};
use rand::{thread_rng, Rng};

const WINDOW_X: u32 = 1000;
const WINDOW_Y: u32 = 1000;
const SPIN_WIDTH_X: u32 = 1;
const SPIN_WIDTH_Y: u32 = 1;
const BETA_START: f32 = 0.1 * 0.440686793509772;
const BETA_END: f32 = 1.0 * 0.440686793509772; // (2.).sqrt().ln_1p() / 2.;
const N_STEPS: u32 = 100;

struct Model {
    _window: window::Id,
    w_x: u32,
    w_y: u32,
    n_x: usize,
    n_y: usize,
    down_rgba: [u8; 4],
    up_rgba: [u8; 4],
    a: Array<i8, Ix2>,
    rng: rand::rngs::ThreadRng,
    beta: f32,
    beta_delta: f32,
    texture: wgpu::Texture,
}

fn model(app: &App) -> Model {
    let _window = app
        .new_window()
        .size(WINDOW_X, WINDOW_Y)
        .view(view)
        .build()
        .unwrap();
    let rng = thread_rng();
    let beta: f32 = BETA_START;
    let beta_delta: f32 = (BETA_END - BETA_START) / N_STEPS as f32;
    let window = app.main_window();
    let wh = window.rect();
    let w_x: u32 = SPIN_WIDTH_X;
    let w_y: u32 = SPIN_WIDTH_Y;
    let n_x: usize = wh.w() as usize / w_x as usize;
    let n_y: usize = wh.h() as usize / w_y as usize;

    let down_rgba: [u8; 4] = [0, 0, 0, u8::MAX];
    let up_rgba: [u8; 4] = [u8::MAX, u8::MAX, u8::MAX, u8::MAX];

    println!("w_x {}, w_y {}, n_x {}, n_y {}", w_x, w_y, n_x, n_y);
    let a = Array::<i8, Ix2>::ones((n_x, n_y).f());
    let texture = wgpu::TextureBuilder::new()
        .size([wh.w() as u32, wh.h() as u32])
        .format(wgpu::TextureFormat::Rgba8Unorm)
        .usage(wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING)
        .build(window.device());

    Model {
        _window,
        w_x,
        w_y,
        n_x,
        n_y,
        a,
        down_rgba,
        up_rgba,
        rng,
        beta,
        beta_delta,
        texture,
    }
}

fn update(_app: &App, _model: &mut Model, _update: Update) {
    let n_x = _model.n_x;
    let n_y = _model.n_y;
    let beta = _model.beta;
    for _ in 0..(n_x * n_y) {
        let i: usize = _model.rng.gen_range(0..n_x);
        let j: usize = _model.rng.gen_range(0..n_y);

        // compute energy of current state and new state
        let old_energy = -_model.a[[i, j]]
            * (_model.a[[(i + n_x - 1) % n_x, j]]
                + _model.a[[(i + 1) % n_x, j]]
                + _model.a[[i, (j + n_y - 1) % n_y]]
                + _model.a[[i, (j + 1) % n_y]]);
        let new_energy = -old_energy;

        // flip a coin
        if _model.rng.gen::<f32>() < (beta * (old_energy - new_energy) as f32).exp() {
            _model.a[[i, j]] *= -1;
        }
    }

    // increment beta
    if _model.beta > BETA_END || _model.beta < BETA_START {
        _model.beta_delta *= -1.;
    }
    _model.beta += _model.beta_delta;
    println!("beta {}", _model.beta);
}

fn get_rgba(pixel_x: usize, pixel_y: usize, _model: &Model) -> [u8; 4] {
    let i: usize = (pixel_x / _model.w_x as usize) % _model.n_x;
    let j: usize = (pixel_y / _model.w_y as usize) % _model.n_y;
    //println!("pixel_x {}, pixel_y {}, i {}, j {}", pixel_x, pixel_y, i, j);
    let val = _model.a[[i, j]];
    if val == -1 {
        _model.down_rgba
    } else {
        _model.up_rgba
    }
}

fn view(app: &App, _model: &Model, frame: Frame) {
    frame.clear(WHITE);

    let wh = app.window_rect().wh();
    let image = image::ImageBuffer::from_fn(wh.x as u32, wh.y as u32, |i, j| {
        let rgba = get_rgba(i as usize, j as usize, _model);
        nannou::image::Rgba(rgba)
    });

    let flat_samples = image.as_flat_samples();
    _model.texture.upload_data(
        app.main_window().device(),
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
