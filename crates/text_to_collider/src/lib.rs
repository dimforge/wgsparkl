// SPDX-License-Identifier: MIT OR Apache-2.0

use cosmic_text::{
    Attrs, Buffer, Color, Edit, Editor, Family, FontSystem, Metrics, Shaping, SwashCache,
};
use parry3d::{
    na::{Vector2, Vector3},
    shape::Cuboid,
};

pub fn get_rects_for(string: &str) -> Vec<Rect> {
    let width = 512f32;
    let height = 256f32;

    let mut font_system = FontSystem::new();
    let mut swash_cache = SwashCache::new();

    let metrics = Metrics::new(32.0, 44.0);
    let mut editor = Editor::new(Buffer::new_empty(metrics.scale(1f32)));
    let mut editor = editor.borrow_with(&mut font_system);

    editor.with_buffer_mut(|buffer| {
        buffer.set_size(Some(width), Some(height));
        let attrs = Attrs::new();
        let comic_attrs = attrs.family(Family::Name("Comic Neue"));

        let spans: &[(&str, Attrs)] =
            &[(string, comic_attrs.metrics(Metrics::relative(64.0, 1.2)))];

        buffer.set_rich_text(spans.iter().copied(), comic_attrs, Shaping::Advanced);
    });

    editor.shape_as_needed(true);

    let font_color = Color::rgb(0xFF, 0xFF, 0xFF);
    let cursor_color = Color(0);
    let selection_color = Color::rgba(0xFF, 0xFF, 0xFF, 0x33);
    let selected_text_color = Color::rgb(0xA0, 0xA0, 0xFF);
    let mut rects = Vec::new();
    editor.draw(
        &mut swash_cache,
        font_color,
        cursor_color,
        selection_color,
        selected_text_color,
        |x, y, lw, lh, color| {
            if color.a() == 0 {
                return;
            }
            rects.push(Rect {
                x,
                // invert y for bevy.
                y: -y,
                width: lw,
                height: lh,
                color,
            })
        },
    );
    rects
}

#[derive(Debug)]
pub struct Rect {
    /// Left of the rect
    pub x: i32,
    /// Top of the rect
    pub y: i32,
    /// Width of the rect
    pub width: u32,
    /// Height of the rect
    pub height: u32,
    /// Color of the rect
    pub color: Color,
}

impl Rect {
    pub fn get_center(&self) -> (f32, f32) {
        (
            self.x as f32 - self.width as f32 / 2f32,
            self.y as f32 + self.height as f32 / 2f32,
        )
    }
    pub fn to_cuboid(&self, half_extent_z: f32) -> (Cuboid, Vector2<f32>, Color) {
        let center = self.get_center();
        return (
            Cuboid::new(Vector3::new(
                self.width as f32 / 2f32,
                self.height as f32 / 2f32,
                half_extent_z,
            )),
            Vector2::new(center.0, center.1),
            self.color,
        );
    }
}
