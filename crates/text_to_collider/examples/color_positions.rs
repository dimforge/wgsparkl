use std::fs;

use bevy::math::Vec2;
use cosmic_text::Color;
use parry3d::{na::Vector2, shape::Cuboid};
use text_to_collider::{get_rects_for, Rect, HEIGHT, WIDTH};

fn main() {
    let rects = get_rects_for("Wgsparkl 🌊✨");
    let positions = std::fs::read("particles.bin")
        .expect("Could not read particles.bin, generate it by running example sand3.");
    let positions = rkyv::from_bytes::<Vec<[f32; 3]>, rkyv::rancor::Error>(&positions).unwrap();
    let detection_radius = 2f32;
    let colors = positions
        .iter()
        .map(|p| {
            // dismiss the y coordinate
            let pos_to_check =
                Vec2::new(p[0], -p[2] + 20f32) * 8f32 + Vec2::new(WIDTH, 0f32) / 2f32;
            if let Some(c) = rects.iter().find(|c| {
                let c1 = Vec2::new(c.x as f32, c.y as f32);
                c1.distance_squared(pos_to_check) < detection_radius
            }) {
                let color = c.color.as_rgba();
                [color[0], color[1], color[2]]
            } else {
                [0, 55, 200]
            }
        })
        .collect::<Vec<_>>();
    let particle_bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&colors).unwrap();
    std::fs::write("particles_colors.bin", &particle_bytes).unwrap();
}
