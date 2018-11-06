use RelativeTimer;
use hoppinworlddata::PlayerStats;
use amethyst_rhusics::rhusics_core::physics3d::Velocity3;
use amethyst_extra::Jump;
use amethyst::ui::{UiTransform, UiText};
use amethyst::core::cgmath::{Vector3, InnerSpace};
use hoppinworldruntime::{PlayerTag,RuntimeProgress};
use avg_float_to_string;
pub use amethyst::ecs::{System, Read,ReadStorage, WriteStorage, Join};

const DISPLAY_SPEED_MULTIPLIER: f32 = 50.0;

/// Very game dependent.
pub struct UiUpdaterSystem;

impl<'a> System<'a> for UiUpdaterSystem {
    type SystemData = (
        Read<'a, RelativeTimer>,
        Read<'a, PlayerStats>,
        ReadStorage<'a, Velocity3<f32>>,
        ReadStorage<'a, Jump>,
        ReadStorage<'a, UiTransform>,
        WriteStorage<'a, UiText>,
        ReadStorage<'a, PlayerTag>,
        Read<'a, RuntimeProgress>,
    );

fn run(&mut self, (timer, _stat, velocities, _jumps, ui_transforms, mut texts, players, runtime_progress): Self::SystemData){
        for (ui_transform, mut text) in (&ui_transforms, &mut texts).join() {
            match &*ui_transform.id {
                "timer" => {
                    text.text = timer.get_text();
                }
                "pb" => {}
                "wr" => {}
                "segment" => {
                    text.text = runtime_progress.current_segment.to_string();
                }
                "speed" => {
                    for (_, velocity) in (&players, &velocities).join() {
                        let vel = velocity.linear();
                        let vel_flat = Vector3::new(vel.x, 0.0, vel.z);
                        let mag = vel_flat.magnitude() * DISPLAY_SPEED_MULTIPLIER;

                        text.text = avg_float_to_string(mag, 1);
                    }
                }
                _ => {}
            }
        }
    }
}