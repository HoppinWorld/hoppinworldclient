use amethyst_extra::nphysics_ecs::DynamicBody;
use RelativeTimer;
use hoppinworlddata::PlayerStats;
use amethyst_extra::Jump;
use amethyst::ui::{UiTransform, UiText};
use amethyst::core::nalgebra::{Vector3};
use hoppinworldruntime::{PlayerTag,RuntimeProgress};
use sec_to_display;
pub use amethyst::ecs::{System, Read,ReadStorage, WriteStorage, Join};

const DISPLAY_SPEED_MULTIPLIER: f32 = 50.0;

/// Very game dependent.
pub struct UiUpdaterSystem;

impl<'a> System<'a> for UiUpdaterSystem {
    type SystemData = (
        Read<'a, RelativeTimer>,
        Read<'a, PlayerStats>,
        ReadStorage<'a, DynamicBody>,
        ReadStorage<'a, Jump>,
        ReadStorage<'a, UiTransform>,
        WriteStorage<'a, UiText>,
        ReadStorage<'a, PlayerTag>,
        Read<'a, RuntimeProgress>,
    );

fn run(&mut self, (timer, _stat, rigid_bodies, _jumps, ui_transforms, mut texts, players, runtime_progress): Self::SystemData){
        for (ui_transform, mut text) in (&ui_transforms, &mut texts).join() {
            match &*ui_transform.id {
                "timer" => {
                    text.text = timer.get_text(3);
                }
                "pb" => {}
                "wr" => {}
                "segment" => {
                    text.text = runtime_progress.current_segment.to_string();
                }
                "speed" => {
                    for (_, rb) in (&players, &rigid_bodies).join() {
                        if let DynamicBody::RigidBody(ref rb) = &rb {
                            let vel = rb.velocity.linear;
                            let vel_flat = Vector3::new(vel.x, 0.0, vel.z);
                            let mag = vel_flat.magnitude() * DISPLAY_SPEED_MULTIPLIER;

                            text.text = sec_to_display(mag.into(), 1);
                        }
                    }
                }
                _ => {}
            }
        }
    }
}