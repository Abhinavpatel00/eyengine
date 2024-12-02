mod camera;
mod camera_controller;
mod instance;
mod state;
mod texture;
mod vertex;

use pollster::FutureExt;
use state::StateApplication;
use winit::event_loop::EventLoop;

pub async fn run() {
    let event_loop = EventLoop::new().unwrap();
    let mut window_state = StateApplication::new();
    let _ = event_loop.run_app(&mut window_state);
}

fn main() {
    pollster::block_on(run());
}

