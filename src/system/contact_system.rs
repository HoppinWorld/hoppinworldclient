

use amethyst_rhusics::rhusics_core::physics3d::Velocity3;
use RelativeTimer;
use amethyst::core::{Transform, Time};
use hoppinworldruntime::{ObjectType, RuntimeProgress, CustomStateEvent, PlayerTag};
use amethyst_extra::BhopMovement3D;
use amethyst_rhusics::rhusics_core::{NextFrame, ContactEvent, Pose};
use amethyst_rhusics::rhusics_core::collide3d::BodyPose3;
use amethyst::shrev::{ReaderId, EventChannel};
use amethyst::ecs::{Resources, Entity, System, Entities, ReadStorage, WriteStorage, Read, Write, Join, SystemData};
use amethyst::core::cgmath::{Point3, Vector2, Vector3, InnerSpace};

/// Very game dependent.
/// Don't try to make that generic.
#[derive(Default)]
pub struct ContactSystem {
    contact_reader: Option<ReaderId<ContactEvent<Entity, Point3<f32>>>>,
}

impl<'a> System<'a> for ContactSystem {
    type SystemData = (
        Entities<'a>,
        ReadStorage<'a, Transform>,
        Read<'a, EventChannel<ContactEvent<Entity, Point3<f32>>>>,
        Write<'a, RelativeTimer>,
        Read<'a, Time>,
        ReadStorage<'a, ObjectType>,
        ReadStorage<'a, PlayerTag>,
        ReadStorage<'a, BhopMovement3D>,
        WriteStorage<'a, NextFrame<Velocity3<f32>>>,
        WriteStorage<'a, NextFrame<BodyPose3<f32>>>,
        Write<'a, EventChannel<CustomStateEvent>>,
        Write<'a, RuntimeProgress>,
    );

    fn run(
        &mut self,
        (
            entities,
            transforms,
            contacts,
            mut timer,
            time,
            object_types,
            players,
            bhop_movements,
            mut velocities,
            mut body_poses,
            mut state_eventchannel,
            mut runtime_progress,
        ): Self::SystemData,
    ) {
        for contact in contacts.read(&mut self.contact_reader.as_mut().unwrap()) {
            //info!("Collision: {:?}",contact);
            let type1 = object_types.get(contact.bodies.0);
            let type2 = object_types.get(contact.bodies.1);

            if type1.is_none() || type2.is_none() {
                continue;
            }
            let type1 = type1.unwrap();
            let type2 = type2.unwrap();

            let (_player, other, player_entity) = if *type1 == ObjectType::Player {
                //(contact.bodies.0,contact.bodies.1)
                (type1, type2, contact.bodies.0)
            } else if *type2 == ObjectType::Player {
                //(contact.bodies.1,contact.bodies.0)
                (type2, type1, contact.bodies.1)
            } else {
                continue;
            };

            match other {
                ObjectType::StartZone => {
                    if runtime_progress.current_segment == 1 {
                        timer.start(time.absolute_time_seconds());
                    }
                    // Also limit player velocity while touching the StartZone to prevent any early starts.
                    // Not sure if this should go into state or not. Since it is heavily related to gameplay I'll put it here.
                    for (entity, _, movement, mut velocity) in
                        (&*entities, &players, &bhop_movements, &mut velocities).join()
                    {
                        if entity == player_entity {
                            let max_vel = movement.max_velocity_ground;
                            let cur_vel3 = *velocity.value.linear();
                            let mut cur_vel_flat = Vector2::new(cur_vel3.x, cur_vel3.z);
                            let cur_vel_flat_mag = cur_vel_flat.magnitude();
                            if cur_vel_flat_mag >= max_vel {
                                cur_vel_flat = cur_vel_flat.normalize() * max_vel;
                                velocity.value.set_linear(Vector3::new(
                                    cur_vel_flat.x,
                                    cur_vel3.y,
                                    cur_vel_flat.y,
                                ))
                            }
                        }
                    }

                    info!("start zone!");
                }
                ObjectType::EndZone => {
                    timer.stop();
                    info!("Finished! time: {:?}", timer.duration());
                    let id = runtime_progress.segment_count as usize;
                    runtime_progress.segment_times[id-1] = timer.duration();
                    state_eventchannel.single_write(CustomStateEvent::MapFinished);
                }
                ObjectType::KillZone => {
                    info!("you are ded!");
                    let seg = runtime_progress.current_segment;
                    let pos = if seg == 1 {
                        // To start zone
                        (&transforms, &object_types).join().filter(|(_,obj)| **obj == ObjectType::StartZone).map(|(tr,_)| tr.translation).next().unwrap()
                    } else {
                        // To last checkpoint

                        // Find checkpoint corresponding to the current segment in progress
                        (&transforms, &object_types).join().filter(|(_,obj)| {
                        	if let ObjectType::SegmentZone(s) = **obj {
                        		s == seg - 1
                        	} else {
                        		false
                        	}
                        }).map(|(tr,_)| tr.translation).next().unwrap()
                    };

                    // Move the player
                    let mut body_pose = (&players, &mut body_poses).join().map(|t| t.1).next().unwrap();
                    let pos = Point3::new(pos.x, pos.y, pos.z);
                    body_pose.value.set_position(pos);
                }
                ObjectType::SegmentZone(id) => {
                    if *id >= runtime_progress.current_segment {
                        runtime_progress.segment_times[(*id-1) as usize] = timer.duration();
                        runtime_progress.current_segment = *id + 1;
                    }
                    info!("segment done");
                }
                _ => {}
            }
        }
    }

    fn setup(&mut self, res: &mut Resources) {
        Self::SystemData::setup(res);
        self.contact_reader = Some(
            res.fetch_mut::<EventChannel<ContactEvent<Entity, Point3<f32>>>>()
                .register_reader(),
        );
    }
}