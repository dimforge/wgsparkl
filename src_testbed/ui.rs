use crate::prep_vertex_buffer::RenderMode;
use crate::startup::RigidParticlesTag;
use crate::{AppState, PhysicsContext, RunState, SceneInits, Timestamps};
use bevy::prelude::*;
use bevy::render::renderer::RenderQueue;
use bevy_egui::egui::{CollapsingHeader, Slider};
use bevy_egui::{egui, EguiContexts};
use nalgebra::vector;
use wgsparkl::solver::SimulationParams;

pub fn update_ui(
    mut commands: Commands,
    mut ui_context: EguiContexts,
    physics: ResMut<PhysicsContext>,
    mut app_state: ResMut<AppState>,
    scenes: Res<SceneInits>,
    timings: Res<Timestamps>,
    queue: Res<RenderQueue>,
    mut rigid_particles: Query<&mut Visibility, With<RigidParticlesTag>>,
) {
    egui::Window::new("Parameters").show(ui_context.ctx_mut(), |ui| {
        let mut changed = false;
        egui::ComboBox::from_label("selected sample")
            .selected_text(&scenes.scenes[app_state.selected_scene].0)
            .show_ui(ui, |ui| {
                for (i, (name, _)) in scenes.scenes.iter().enumerate() {
                    changed = ui
                        .selectable_value(&mut app_state.selected_scene, i, name)
                        .changed()
                        || changed;
                }
            });
        if changed {
            scenes.init_scene(&mut commands, app_state.selected_scene);
            app_state.restarting = false;
        }

        let mut changed = false;
        egui::ComboBox::from_label("render mode")
            .selected_text(RenderMode::from_u32(app_state.render_config.mode).text())
            .show_ui(ui, |ui| {
                for i in 0..6 {
                    changed = ui
                        .selectable_value(
                            &mut app_state.render_config.mode,
                            i,
                            RenderMode::from_u32(i).text(),
                        )
                        .changed()
                        || changed;
                }
            });

        if changed {
            queue.write_buffer(
                app_state.gpu_render_config.buffer.buffer(),
                0,
                bytemuck::bytes_of(&app_state.render_config.mode),
            );
            queue.submit([]);
        }

        let mut sim_params_changed = false;
        sim_params_changed = ui
            .add(Slider::new(&mut app_state.num_substeps, 1..=200).text("substeps"))
            .changed()
            || sim_params_changed;
        sim_params_changed = ui
            .add(Slider::new(&mut app_state.gravity_factor, 0.0..=10.0).text("gravity factor"))
            .changed()
            || sim_params_changed;

        if ui
            .checkbox(&mut app_state.show_rigid_particles, "show rigid_particles")
            .changed()
        {
            for mut visibility in rigid_particles.iter_mut() {
                if app_state.show_rigid_particles {
                    *visibility = Visibility::Inherited;
                } else {
                    *visibility = Visibility::Hidden;
                }
            }
        }

        #[cfg(feature = "dim2")]
        let gravity = vector![0.0, -9.81];
        #[cfg(feature = "dim3")]
        let gravity = vector![0.0, -9.81, 0.0];

        if sim_params_changed {
            let new_params = SimulationParams {
                gravity: gravity * app_state.gravity_factor,
                dt: (1.0 / 60.0) / (app_state.num_substeps as f32),
                #[cfg(feature = "dim2")]
                padding: 0.0,
            };
            queue.write_buffer(
                physics.data.sim_params.params.buffer(),
                0,
                bytemuck::bytes_of(&new_params),
            );
            queue.submit([]);
        }

        ui.label(format!("Particle count: {}", physics.particles.len()));
        ui.label(format!(
            "Rigid particle count: {}",
            physics.data.rigid_particles.len()
        ));

        CollapsingHeader::new(format!("GPU runtime: {:.3}ms", timings.total_time()))
            .id_salt("GPU runtimes")
            .show(ui, |ui| {
                ui.label(format!(
                    "Rigid update: {:.3}ms",
                    timings.update_rigid_particles
                ));
                ui.label(format!("Grid sort: {:.3}ms", timings.grid_sort));
                ui.label(format!("CDF Grid update: {:.3}ms", timings.grid_update_cdf));
                ui.label(format!("CDF P2G: {:.3}ms", timings.p2g_cdf));
                ui.label(format!("CDF G2P: {:.3}ms", timings.g2p_cdf));
                ui.label(format!("P2G: {:.3}ms", timings.p2g));
                ui.label(format!("Grid update: {:.3}ms", timings.grid_update));
                ui.label(format!("G2P: {:.3}ms", timings.g2p));
                ui.label(format!(
                    "Particles update: {:.3}ms",
                    timings.particles_update
                ));
                ui.label(format!(
                    "Integrate bodies: {:.3}ms",
                    timings.integrate_bodies
                ));
            });

        ui.horizontal(|ui| {
            let label = if app_state.run_state == RunState::Paused {
                "Run"
            } else {
                "Pause"
            };

            if ui.button(label).clicked() {
                if app_state.run_state == RunState::Paused {
                    app_state.run_state = RunState::Running
                } else {
                    app_state.run_state = RunState::Paused
                }
            }

            if ui.button("Step").clicked() {
                app_state.run_state = RunState::Step;
            }

            if ui.button("Restart").clicked() {
                scenes.init_scene(&mut commands, app_state.selected_scene);
                app_state.restarting = true;
            }
        });
    });
}
