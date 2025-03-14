use bevy::{input::common_conditions::input_toggle_active, prelude::*};
use cosmic_text::Color;
use parry3d::{
    bounding_volume::BoundingVolume,
    math::Point,
    na::{Isometry3, Point2, SVector, UnitQuaternion, Vector2},
    query::PointQuery,
    shape::{Ball, Cuboid},
};
use text_to_collider::{get_rects_for, Rect};

fn main() {
    let rects = get_rects_for("Wgsparkl 🌊✨");

    App::new()
        .add_plugins(DefaultPlugins)
        .insert_resource(Cuboids(rects.iter().map(|r| r.to_cuboid(2f32)).collect()))
        .insert_resource(RectsStorage(rects))
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            show_rects.run_if(input_toggle_active(true, KeyCode::Space)),
        )
        .add_systems(Update, show_cuboids)
        .run();
}

#[derive(Resource)]
pub struct RectsStorage(pub Vec<Rect>);

#[derive(Resource)]
pub struct Cuboids(pub Vec<(Cuboid, Vector2<f32>, Color)>);

pub fn setup(mut commands: Commands) {
    commands.spawn((Camera::default(), Camera2d));
}

pub fn show_rects(mut g: Gizmos, rects: Res<RectsStorage>) {
    for r in rects.0.iter() {
        let color = r.color.as_rgba();
        g.rect(
            Isometry3d::from_translation(Vec3::new(r.x as f32, r.y as f32, 0f32)),
            Vec2::new(r.width as f32, r.height as f32),
            bevy::prelude::Color::srgba_u8(color[0], color[1], color[2], color[3]),
        );
    }
}

pub fn show_cuboids(time: Res<Time>, mut g: Gizmos, cuboids: Res<Cuboids>) {
    let detection_radius = 1f32;
    for i in 0..8000 {
        let t = ((time.elapsed_secs()) + i as f32 * 10.5) * 0.1f32;
        let pos_to_check = Vec2::new(
            (((t * (0.5 + (i + 1) as f32 / 5000f32) * 0.7f32).cos() + 1f32) / 2f32) * 500f32,
            (((t * (0.5 + (i + 1) as f32 / 5000f32) * 0.71f32).cos() + 1f32) / 2f32) * -84f32,
        );
        if let Some(c) = cuboids.0.iter().find(|c| {
            let c1 = Vec2::new(c.1.x, c.1.y);
            c1.distance_squared(pos_to_check) < detection_radius
        }) {
            let color = c.2.as_rgba();
            g.sphere(
                Isometry3d::from_translation(pos_to_check.extend(0f32)),
                detection_radius,
                bevy::prelude::Color::srgba_u8(color[0], color[1], color[2], color[3]),
            );
        } else {
            g.sphere(
                Isometry3d::from_translation(pos_to_check.extend(0f32)),
                detection_radius,
                bevy::prelude::Color::srgba_u8(200, 0, 0, 255),
            );
        }
    }
}
