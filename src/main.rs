extern crate amethyst;
extern crate amethyst_gltf;
extern crate amethyst_extra;
extern crate amethyst_rhusics;
#[macro_use]
extern crate serde;
#[macro_use]
extern crate log;
extern crate winit;
extern crate partial_function;
#[macro_use]
extern crate derive_new;
#[macro_use]
extern crate specs_derive;

use amethyst::core::timing::duration_to_secs;
use amethyst::assets::{PrefabLoader, PrefabLoaderSystem, RonFormat, AssetPrefab, ProgressCounter, PrefabData, AssetStorage, Handle};
use amethyst::assets::Error as AssetError;
use amethyst::ecs::error::Error as ECSError;
use amethyst::shrev::{ReaderId, EventChannel};
use std::hash::Hash;
use amethyst::ui::*;
use amethyst::controls::{FlyControlBundle, FlyControlTag, FlyMovementSystem,WindowFocus,MouseFocusUpdateSystem,FreeRotationSystem,CursorHideSystem};
use amethyst::core::transform::{Transform, TransformBundle, Parent};
use amethyst::core::{WithNamed,Time, Named, Stopwatch};
use amethyst::ecs::{
    Component, DenseVecStorage, Join, Read, ReadStorage, System, VecStorage, Write, WriteStorage, Resources, SystemData, Entity, Entities, ParJoin,
};
use amethyst::ecs::storage::AntiStorage;
use amethyst::ecs::prelude::ParallelIterator;
use amethyst::input::{InputBundle, InputHandler};
use amethyst::prelude::*;
use amethyst::renderer::*;
use amethyst::utils::scene::BasicScenePrefab;
use amethyst::Error;
use amethyst_gltf::{GltfSceneAsset, GltfSceneFormat, GltfSceneLoaderSystem};

use amethyst::core::cgmath::{Matrix3, One, Point3, Quaternion, Vector3, Zero, Deg, Euler, InnerSpace, Rotation, EuclideanSpace, Rotation3, Vector2, SquareMatrix, Basis3};
use amethyst_rhusics::collision::primitive::{Cuboid, Primitive3, ConvexPolyhedron, Cylinder};
use amethyst_rhusics::collision::{Aabb3,Ray3};
use amethyst_rhusics::rhusics_core::physics3d::{Mass3, Velocity3};
use amethyst_rhusics::rhusics_core::{
    Collider, CollisionMode, CollisionShape, CollisionStrategy, ForceAccumulator, Inertia, Mass,
    Pose, PhysicalEntity, Velocity,Material,NextFrame, ContactEvent,
};
use amethyst_rhusics::rhusics_ecs::physics3d::{BodyPose3,DynamicBoundingVolumeTree3};
use amethyst_rhusics::collision::dbvt::query_ray;
use amethyst_rhusics::rhusics_ecs::{WithPhysics,PhysicalEntityParts};
use amethyst_rhusics::{time_sync, DefaultPhysicsBundle3};
use std::marker::PhantomData;
use winit::DeviceEvent;

use amethyst_extra::*;
use partial_function::*;

type ScenePrefab = BasicScenePrefab<Vec<PosNormTex>>;
type Shape = CollisionShape<Primitive3<f32>, BodyPose3<f32>, Aabb3<f32>, ObjectType>;
type DefaultPhysicalEntityParts<'a, T> = PhysicalEntityParts<'a, Primitive3<f32>,T,Quaternion<f32>,Vector3<f32>,Vector3<f32>,Matrix3<f32>,Aabb3<f32>,BodyPose3<f32>>;
type MyPhysicalEntityParts<'a> = DefaultPhysicalEntityParts<'a, ObjectType>;

#[repr(u8)]
#[derive(Debug, Clone, PartialOrd, PartialEq, Component)]
pub enum ObjectType {
    Scene,
    StartZone,
    EndZone,
    KillZone,
    Player,
}

impl Default for ObjectType {
    fn default() -> Self {
        ObjectType::Scene
    }
}

impl Collider for ObjectType {
    fn should_generate_contacts(&self, other: &ObjectType) -> bool {
    	let ret = (*self == ObjectType::Player && *other == ObjectType::Scene)
        || (*self == ObjectType::Scene && *other == ObjectType::Player);
    	//info!("should_generate_contacts {:?} -> {:?} ret {}", *self, *other, ret);
        true
            //other != self
            //true
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, new)]
struct Gravity {
    pub acceleration: Vector3<f32>,
}

impl Default for Gravity {
    fn default() -> Self {
        Gravity {
            acceleration: Vector3::new(0.0, -1.0, 0.0),
        }
    }
}

struct GravitySystem;

impl<'a> System<'a> for GravitySystem {
    type SystemData = (
    	Read<'a, Time>,
        Read<'a, Gravity>,
        WriteStorage<'a, NextFrame<Velocity3<f32>>>,
    );
    fn run(&mut self, (time, gravity, mut velocities): Self::SystemData) {
        for (mut velocity,) in (&mut velocities,).join() {
        	// Add the acceleration to the velocity.
            let new_vel = velocity.value.linear() + gravity.acceleration * time.delta_seconds();
			velocity.value.set_linear(new_vel);
        }
    }
}

/// The system that manages the fly movement.
/// Generic parameters are the parameters for the InputHandler.
pub struct FpsMovementSystemSimple<A, B> {
    /// The movement speed of the movement in units per second.
    speed: f32,
    /// The name of the input axis to locally move in the x coordinates.
    /// Left and right.
    right_input_axis: Option<A>,
    /// The name of the input axis to locally move in the z coordinates.
    /// Forward and backward. Please note that -z is forward when defining your input configurations.
    forward_input_axis: Option<A>,
    _phantomdata: PhantomData<B>,
}

impl<A, B> FpsMovementSystemSimple<A, B>
where
    A: Send + Sync + Hash + Eq + Clone + 'static,
    B: Send + Sync + Hash + Eq + Clone + 'static,
{
    pub fn new(
        speed: f32,
        right_input_axis: Option<A>,
        forward_input_axis: Option<A>,
    ) -> Self {
        FpsMovementSystemSimple {
            speed,
            right_input_axis,
            forward_input_axis,
            _phantomdata: PhantomData,
        }
    }
}

impl<'a, A, B> System<'a> for FpsMovementSystemSimple<A, B>
where
    A: Send + Sync + Hash + Eq + Clone + 'static,
    B: Send + Sync + Hash + Eq + Clone + 'static,
{
    type SystemData = (
        Read<'a, Time>,
        WriteStorage<'a, Transform>,
        Read<'a, InputHandler<A, B>>,
        ReadStorage<'a, FlyControlTag>,
        WriteStorage<'a, ForceAccumulator<Vector3<f32>, Vector3<f32>>>,
    );

    fn run(&mut self, (time, transforms, input, tags, mut forces): Self::SystemData) {
        let x = get_input_axis_simple(&self.right_input_axis, &input);
        let z = get_input_axis_simple(&self.forward_input_axis, &input);

        let dir = Vector3::new(x, 0.0, z);
        if dir.magnitude() != 0.0 {
        	for (transform, _, mut force) in (&transforms, &tags, &mut forces).join() {
	            let mut dir = transform.rotation * dir;
	            dir = dir.normalize();
	            force.add_force(dir * self.speed * time.delta_seconds());
        	}
        }
    }
}



#[derive(Debug,Clone,Default,Serialize,Deserialize,Component)]
pub struct RotationControl{
    pub mouse_accum_x: f32,
    pub mouse_accum_y: f32,
}

#[derive(Debug,Clone,Default,Serialize,Deserialize,new,Component)]
pub struct Grounded{
	#[new(value = "false")]
	 pub ground: bool,
	 #[new(default)]
	 pub since: f64,
	 pub distance_check: f32,
}

/// The system that manages the view rotation.
/// Controlled by the mouse.
pub struct FPSRotationRhusicsSystem<A, B> {
    sensitivity_x: f32,
    sensitivity_y: f32,
    _marker1: PhantomData<A>,
    _marker2: PhantomData<B>,
    event_reader: Option<ReaderId<Event>>,
}

impl<A, B> FPSRotationRhusicsSystem<A, B> {
    pub fn new(sensitivity_x: f32, sensitivity_y: f32) -> Self {
        FPSRotationRhusicsSystem {
            sensitivity_x,
            sensitivity_y,
            _marker1: PhantomData,
            _marker2: PhantomData,
            event_reader: None,
        }
    }
}

impl<'a, A, B> System<'a> for FPSRotationRhusicsSystem<A, B>
where
    A: Send + Sync + Hash + Eq + Clone + 'static,
    B: Send + Sync + Hash + Eq + Clone + 'static,
{
    type SystemData = (
        Read<'a, EventChannel<Event>>,
        WriteStorage<'a, Transform>,
        WriteStorage<'a, BodyPose3<f32>>,
        WriteStorage<'a, NextFrame<BodyPose3<f32>>>,
        WriteStorage<'a, RotationControl>,
        ReadStorage<'a, FlyControlTag>,
        Read<'a, WindowFocus>,
    );

    fn run(&mut self, (events, mut transforms, mut body_poses, mut next_body_poses, mut rotation_controls, fly_controls,focus): Self::SystemData) {
        let focused = focus.is_focused;
        for event in events.read(&mut self.event_reader.as_mut().unwrap()) {
            if focused {
                match *event {
                    Event::DeviceEvent { ref event, .. } => match *event {
                        DeviceEvent::MouseMotion { delta: (x, y) } => {
                            for (mut transform, mut rotation_control) in (&mut transforms, &mut rotation_controls).join() {
                            	rotation_control.mouse_accum_x -= x as f32 * self.sensitivity_x;
                            	rotation_control.mouse_accum_y += y as f32 * self.sensitivity_y;
                            	// Limit maximum vertical angle to prevent locking the quaternion and/or going upside down.
                            	rotation_control.mouse_accum_y = rotation_control.mouse_accum_y.max(-89.5).min(89.5);

                            	transform.rotation = Quaternion::from_angle_x(Deg(-rotation_control.mouse_accum_y));

                            	for (mut body_pose, mut next_body_pose, _) in (&mut body_poses, &mut next_body_poses, &fly_controls).join() {
                            		body_pose.set_rotation(Quaternion::from_angle_y(Deg(rotation_control.mouse_accum_x)));
                            		next_body_pose.value.set_rotation(Quaternion::from_angle_y(Deg(rotation_control.mouse_accum_x)));
                            	}
                            }
                        }
                        _ => (),
                    },
                    _ => (),
                }
            }
        }
    }

    fn setup(&mut self, res: &mut Resources) {
        Self::SystemData::setup(res);
        self.event_reader = Some(res.fetch_mut::<EventChannel<Event>>().register_reader());
    }
}

pub struct GroundCheckerSystem;

impl<'a> System<'a> for GroundCheckerSystem {
    type SystemData = (
    	Entities<'a>,
        ReadStorage<'a, Transform>,
        WriteStorage<'a, Grounded>,
        Read<'a, DynamicBoundingVolumeTree3<f32>>,
        Read<'a, Time>,
    );

    fn run(&mut self, (entities, transforms, mut grounded, tree, time): Self::SystemData) {
    	let down = -Vector3::unit_y();
    	for (entity, transform, mut grounded) in (&*entities, &transforms, &mut grounded).join() {
    		let mut ground = false;

    		let ray = Ray3::new(Point3::from_vec(transform.translation), down);
		    for (v, p) in query_ray(&*tree, ray) {
		    	if v.value != entity {
		    		if (transform.translation - Vector3::new(p.x,p.y,p.z)).magnitude() <= grounded.distance_check {
			    		ground = true;
			            //info!("hit bounding volume of {:?} at point {:?}", v.value, p);
		            }
		        }
		    }

		    if ground && !grounded.ground {
		    	// Just grounded
		    	grounded.since = time.absolute_time_seconds();
		    }
		    grounded.ground = ground;
    	}
    }
}

#[derive(Default, Component, new)]
pub struct Jump{
	pub absolute: bool,
	 pub check_ground: bool,
	 pub jump_force: f32,
	 pub auto_jump: bool,
	 #[new(value = "0.1")]
	 pub jump_cooldown: f64,
	 #[new(value = "0.1")]
	 pub input_cooldown: f64,
	 /// Multiplier. Time can go in the negatives.
	 #[new(default)]
	 pub jump_timing_boost: Option<PartialFunction<f64,f32>>,
	 #[new(default)]
	 pub last_jump: f64,
	 #[new(default)]
	 pub last_jump_offset: f64,
}

#[derive(Default)]
pub struct JumpSystem{
	/// The last time the system considered a valid jump input.
	last_logical_press: f64,
	/// Was the jump key pressed last frame?
    input_hold: bool,
    /// The last time we physically pressed the jump key.
    last_physical_press: f64,
}

impl<'a> System<'a> for JumpSystem {
    type SystemData = (
    	Entities<'a>,
        ReadStorage<'a, Grounded>,
        WriteStorage<'a, Jump>,
        Read<'a, Time>,
        Read<'a, InputHandler<String, String>>,
        WriteStorage<'a, ForceAccumulator<Vector3<f32>, Vector3<f32>>>,
        WriteStorage<'a, NextFrame<Velocity3<f32>>>,
    );

    fn run(&mut self, (entities, grounded, mut jumps, time, input, mut forces, mut velocities): Self::SystemData) {
    	if let Some(true) = input.action_is_down("jump") {
    		if !self.input_hold {
    			// We just started pressing the key. Registering time.
    			self.last_physical_press = time.absolute_time_seconds();
    			self.input_hold = true;
    		}

	    	for (entity, mut jump, mut force, mut velocity) in (&*entities, &mut jumps, &mut forces, &mut velocities).join() {
	    		// Holding the jump key on a non-auto jump controller.
	    		if self.input_hold && !jump.auto_jump {
	    			continue;
	    		}

	    		// The last time we jumped wasn't long enough ago
	    		if time.absolute_time_seconds() - self.last_logical_press < jump.input_cooldown {
	    			continue;
	    		}
	    		self.last_logical_press = time.absolute_time_seconds();

	    		// If we need to check for it, verify that we are on the ground.
	    		let mut grounded_since = time.absolute_time_seconds();
	    		if jump.check_ground {
	    			if let Some(ground) = grounded.get(entity) {
	    				if !ground.ground {
	    					continue;
	    				}
	    				grounded_since = ground.since;
	    			} else {
	    				continue;
	    			}
	    		}

	    		if time.absolute_time_seconds() - jump.last_jump > jump.jump_cooldown {
	    			// Jump!
		    		jump.last_jump = time.absolute_time_seconds();
		    		// Offset for jump. Positive = time when we jumped AFTER we hit the ground.
		    		jump.last_jump_offset = grounded_since - self.last_physical_press;

		    		let multiplier = if let Some(ref curve) = jump.jump_timing_boost {
		    			curve.eval(jump.last_jump_offset).unwrap_or(1.0)
		    		} else {
		    			1.0
		    		};

		    		if jump.absolute {
		    			force.add_force(Vector3::<f32>::unit_y() * jump.jump_force * multiplier);
		    		} else {
		    			let (x,z) = {
		    				let v = velocity.value.linear();
		    				(v.x, v.z)
		    			};
		    			velocity.value.set_linear(Vector3::new(x, jump.jump_force, z));
		    		}
	    		}
	    	}
        } else {
        	// The jump key was released.
        	self.input_hold = false;
        }
    }
}

#[derive(Deserialize, Serialize)]
pub struct HoppinMapPrefabData {
	name: String,
	map: AssetPrefab<GltfSceneAsset, GltfSceneFormat>,
}

impl<'a> PrefabData<'a> for HoppinMapPrefabData {
    type SystemData = (
        <AssetPrefab<GltfSceneAsset, GltfSceneFormat> as PrefabData<'a>>::SystemData,
    );
    type Result = ();

    fn load_prefab(
        &self,
        entity: Entity,
        system_data: &mut Self::SystemData,
        entities: &[Entity],
    ) -> Result<(), ECSError> {
        let (
            ref mut gltfs,
        ) = system_data;
        self.map.load_prefab(entity, gltfs, entities)?;
        Ok(())
    }

    fn trigger_sub_loading(
        &mut self,
        progress: &mut ProgressCounter,
        system_data: &mut Self::SystemData,
    ) -> Result<bool, ECSError> {
        let (ref mut gltfs,) = system_data;
        self.map.trigger_sub_loading(progress, gltfs)
    }
}

/// The settings controlling how the entity controlled by the `BhopMovementSystem` will behave.
/// This is a component that you should add on the entity.
#[derive(Serialize, Deserialize, Debug, Clone, Component, new)]
pub struct BhopMovement3D {
	/// False = Forces, True = Velocity
	pub absolute: bool,
    /// Use world coordinates XYZ.
    #[new(default)]
	pub absolute_axis: bool,
    /// Negates the velocity when pressing the key opposite to the current velocity.
    /// Effectively a way to instantly stop, even at high velocities.
    #[new(default)]
	pub counter_impulse: bool,
    /// Acceleration in unit/s² while on the ground.
    pub accelerate_ground: f32,
    /// Acceleration in unit/s² while in the air.
    pub accelerate_air: f32,
    /// The maximum ground velocity.
    pub max_velocity_ground: f32,
    /// The maximum air velocity.
    pub max_velocity_air: f32,
    /// Enables accelerating over maximumVelocity by airstrafing. Bunnyhop in a nutshell.
	pub allow_projection_acceleration: bool,
}

/// The way friction is applied.
#[derive(Serialize, Deserialize, Copy, Clone, Debug)]
pub enum FrictionMode {
	/// The velocity is reduced by a fixed amount each second (deceleration).
	Linear,
	/// The velocity is reduced by a fraction of the current velocity.
	/// A value of 0.2 means that approximatively 20% of the speed will be lost each second.
	/// Since it is not calculated as an integration but as discrete values, the actual slowdown will vary slightly from case to case.
	Percent,
}

/// Component you add to your entities to apply a ground friction.
/// What the friction field does is dependent on the choosen `FrictionMode`.
#[derive(Serialize,Deserialize,Clone,Debug,Component,new)]
pub struct GroundFriction3D {
	/// The amount of friction speed loss by second.
	pub friction: f32,
	/// The way friction is applied.
	pub friction_mode: FrictionMode,
	/// The time to wait after touching the ground before applying the friction.
	pub ground_time_before_apply: f64,
}

/// The system that manages the first person movements (with added projection acceleration capabilities).
/// Generic parameters are the parameters for the InputHandler.
#[derive(new)]
pub struct BhopMovementSystem<A, B> {
    /// The name of the input axis to locally move in the x coordinates.
    right_input_axis: Option<A>,
    /// The name of the input axis to locally move in the z coordinates.
    forward_input_axis: Option<A>,
    phantom_data: PhantomData<B>,
}

impl<'a, A, B> System<'a> for BhopMovementSystem<A, B>
where
    A: Send + Sync + Hash + Eq + Clone + 'static,
    B: Send + Sync + Hash + Eq + Clone + 'static,
{
    type SystemData = (
        Read<'a, Time>,
        Read<'a, InputHandler<A, B>>,
        ReadStorage<'a, Transform>,
        ReadStorage<'a, BhopMovement3D>,
        ReadStorage<'a, Grounded>,
        WriteStorage<'a, NextFrame<Velocity3<f32>>>,
    );

    fn run(&mut self, (time, input, transforms, movements, groundeds, mut velocities): Self::SystemData) {
    	let x = get_input_axis_simple(&self.right_input_axis, &input);
	    let z = get_input_axis_simple(&self.forward_input_axis, &input);
	    let input = Vector2::new(x, z);

	    if input.magnitude() != 0.0 {
	    	for (transform, movement, grounded, mut velocity) in (&transforms, &movements, &groundeds, &mut velocities).join() {
	    		let (acceleration, max_velocity) = match grounded.ground {
	    			true => (movement.accelerate_ground, movement.max_velocity_ground),
	    			false => (movement.accelerate_air, movement.max_velocity_air),
	    		};

	    		// Global to local coords.
	            let mut relative = SquareMatrix::invert(Basis3::from(transform.rotation).as_ref()).unwrap() * velocity.value.linear();

	            let new_vel_rel = if movement.absolute {
	            	// Absolute = We immediately set the maximum velocity without checking the max speed.
	            	Vector3::new(input.x * acceleration, relative.y, input.y * acceleration)
	            } else {
	            	let mut wish_vel = relative;

	            	if movement.counter_impulse {
	            		wish_vel = counter_impulse(input, wish_vel);
	            	}

	            	wish_vel = accelerate_vector(time.delta_seconds(), input, wish_vel, acceleration, max_velocity);
	            	if !movement.allow_projection_acceleration {
	            		wish_vel = limit_velocity(wish_vel, max_velocity);
	            	}

	            	wish_vel
	            };

	            // Global to local coords;
	            let new_vel = transform.rotation * new_vel_rel;

	            // Assign the new velocity to the player
	            velocity.value.set_linear(new_vel);
	    	}
    	}
    }
}



/// Applies friction (slows the velocity down) according to the `GroundFriction3D` component of your entity.
/// Your entity also needs to have a `Grounded` component (and the `GroundCheckerSystem` added to your dispatcher) to detect the ground.
/// It also needs to have a NextFrame<Velocity3<f32>> component. This is added automatically by rhusics when creating a dynamic physical entity.
pub struct GroundFrictionSystem;

impl<'a> System<'a> for GroundFrictionSystem {
    type SystemData = (
        Read<'a, Time>,
        ReadStorage<'a, Grounded>,
        ReadStorage<'a, GroundFriction3D>,
        WriteStorage<'a, NextFrame<Velocity3<f32>>>,
    );

    fn run(&mut self, (time, groundeds, frictions, mut velocities): Self::SystemData) {
    	fn apply_friction_single(v: f32,friction: f32) -> f32{
	        if v.abs() <= friction {
	            return 0.0;
	        }
        	v - friction
    	}
    	for (grounded, friction, mut velocity) in (&groundeds, &frictions, &mut velocities).join() {
    		if grounded.ground && time.absolute_time_seconds() - grounded.since >= friction.ground_time_before_apply {
    			let (x,y,z) = {
					let v = velocity.value.linear();
					(v.x, v.y, v.z)
				};
	    		match friction.friction_mode {
	    			FrictionMode::Linear => {
						let slowdown = friction.friction * time.delta_seconds();
		                velocity.value.set_linear(Vector3::new(apply_friction_single(x, slowdown),y,apply_friction_single(z, slowdown)));
	    			},
	    			FrictionMode::Percent => {
		                let coef = friction.friction * time.delta_seconds();
		                velocity.value.set_linear(Vector3::new(apply_friction_single(x, x*coef),y,apply_friction_single(z, z*coef)));
	    			},
	    		}
    		}
    	}
    }
}

/// Accelerates the given `relative` vector by the given `acceleration` and `input`.
/// The `maximum_velocity` is only taken into account for the projection of the acceleration vector on the `relative` vector.
/// This allows going over the speed limit by performing what is called a "strafe".
/// If your velocity is forward and have an input accelerating you to the right, the projection of
/// the acceleration vector over your current velocity will be 0. This means that the acceleration vector will be applied fully,
/// even if this makes the resulting vector's magnitude go over `max_velocity`.
pub fn accelerate_vector(delta_time: f32, input: Vector2<f32>, rel: Vector3<f32>, acceleration: f32, max_velocity: f32) -> Vector3<f32> {
	let mut o = rel;
    let input3 = Vector3::new(input.x, 0.0, input.y);
    let rel_flat = Vector3::new(rel.x, 0.0, rel.z);
    if input3.magnitude() > 0.0 {
        let proj = rel_flat.dot(input3.normalize());
        let mut accel_velocity = acceleration * delta_time as f32;
        if proj + accel_velocity > max_velocity {
            accel_velocity = max_velocity - proj;
        }
        if accel_velocity > 0.0 {
            let add_speed = input3 * accel_velocity;
            o += add_speed;
        }
    }
    return o;
}

/// Completely negates the velocity of a specific axis if an input is performed in the opposite direction.
pub fn counter_impulse(input: Vector2<f32>, relative_velocity: Vector3<f32>) -> Vector3<f32> {
    let mut  o = relative_velocity;
    if input.x < 0.0 && relative_velocity.x > 0.001 {
        o = Vector3::new(0.0, relative_velocity.y, relative_velocity.z);
    } else if input.x > 0.0 && relative_velocity.x < -0.001 {
        o = Vector3::new(0.0, relative_velocity.y, relative_velocity.z);
    }
    if input.y < 0.0 && relative_velocity.z > 0.001 {
        o = Vector3::new(relative_velocity.x, relative_velocity.y, 0.0);
    } else if input.y > 0.0 && relative_velocity.z < -0.001 {
        o = Vector3::new(relative_velocity.x, relative_velocity.y, 0.0);
    }
    return o;
}

/// Limits the total velocity so that its magnitude doesn't exceed `maximum_velocity`.
/// If you are using the `accelerate_vector` function, calling this will ensure that air strafing
/// doesn't allow you to go over the maximum velocity, while still keeping fluid controls.
pub fn limit_velocity(vec: Vector3<f32>, maximum_velocity: f32) -> Vector3<f32> {
    let v_flat = Vector2::new(vec.x,vec.z).magnitude();
    if v_flat > maximum_velocity && maximum_velocity != 0.0 {
        let ratio = maximum_velocity / v_flat;
        return Vector3::new(vec.x * ratio,vec.y,vec.z * ratio);
    }
    vec
}

/// Gets the input axis value from the `InputHandler`.
/// If the name is None, it will return the default value of the axis (0.0).
pub fn get_input_axis_simple<A, B>(name: &Option<A>, input: &InputHandler<A, B>) -> f32 
where
    A: Send + Sync + Hash + Eq + Clone + 'static,
    B: Send + Sync + Hash + Eq + Clone + 'static,
{
    name.as_ref()
        .and_then(|ref n| input.axis_value(n))
        .unwrap_or(0.0) as f32
}

#[derive(Default, Component)]
pub struct Timer {
	pub timer: Stopwatch,
}

impl UiAutoText for Timer {
	fn get_text(&self) -> String {
		((duration_to_secs(self.timer.elapsed()) * 1000.0).ceil() / 1000.0).to_string()
	}
}

#[derive(Default)]
pub struct ContactSystem{
	contact_reader: Option<ReaderId<ContactEvent<Entity, Point3<f32>>>>,
}

impl<'a> System<'a> for ContactSystem {
	type SystemData = (
		Read<'a, EventChannel<ContactEvent<Entity, Point3<f32>>>>,
		WriteStorage<'a, Timer>,
		ReadStorage<'a, ObjectType>,
	);

	fn run(&mut self, (contacts, mut timers, object_types): Self::SystemData) {
		for contact in contacts.read(&mut self.contact_reader.as_mut().unwrap()) {
			for (mut timer,) in (&mut timers,).join() {
				//info!("Collision: {:?}",contact);
				let type1 = object_types.get(contact.bodies.0);
				let type2 = object_types.get(contact.bodies.1);

				if type1.is_none() || type2.is_none() {
					continue;
				}
				let type1 = type1.unwrap();
				let type2 = type2.unwrap();

				let (player,other) = if *type1 == ObjectType::Player {
					//(contact.bodies.0,contact.bodies.1)
					(type1, type2)
				} else if *type2 == ObjectType::Player {
					//(contact.bodies.1,contact.bodies.0)
					(type2, type1)
				} else {
					continue;
				};

				match other {
					ObjectType::StartZone => {
						timer.timer.restart();
						info!("start zone!");
					},
					ObjectType::EndZone => {
						timer.timer.stop();
						info!("Finished! time: {:?}", timer.timer);
					},
					ObjectType::KillZone => {
						info!("you are ded!");
					},
					_ => {},
				}
        	}
        }
	}

	fn setup(&mut self, res: &mut Resources) {
        Self::SystemData::setup(res);
        self.contact_reader = Some(
            res.fetch_mut::<EventChannel<ContactEvent<Entity, Point3<f32>>>>().register_reader(),
        );
    }
}


#[derive(Default)]
struct GameState {
    load_progress: Option<ProgressCounter>,
    init_done: bool,
}

impl<'a, 'b> SimpleState<'a, 'b> for GameState {
    fn on_start(&mut self, data: StateData<GameData>) {
    	self.init_done = false;

        let mut pg = ProgressCounter::new();

        let scene_handle = data.world.exec(
            |loader: PrefabLoader<HoppinMapPrefabData>| {
                loader.load(
                    "assets/base/maps/test01.hop",
                    RonFormat,
                    (),
                    &mut pg,
                )
            }
        );
        self.load_progress = Some(pg);

        data.world.create_entity().with(scene_handle).build();

        data.world.add_resource(Gravity::new(Vector3::new(0.0, -2.0, 0.0)));

        data.world.register::<ObjectType>();

        let mut tr = Transform::default();
        tr.translation = [0.0,5.0,0.0].into();

        let movement = BhopMovement3D::new(false, 20.0, 20.0, 2.0, 0.5, true);
        let ground_friction = GroundFriction3D::new(2.0, FrictionMode::Percent, 0.15);

        let player_entity = data.world
            .create_entity()
            .with(FlyControlTag)
            .with(Grounded::new(0.5))
            .with(ObjectType::Player)
            .with(movement)
            .with(ground_friction)
            .with(Jump::new(true, true, 50.0, true))
            .with_dynamic_physical_entity(
            	Shape::new_simple_with_type(
	                CollisionStrategy::FullResolution,
	                CollisionMode::Discrete,
	                Cylinder::new(0.5, 0.2).into(),
	                ObjectType::Player,
	            ),
	            BodyPose3::new(Point3::new(tr.translation.x, tr.translation.y, tr.translation.z), Quaternion::<f32>::one()),
	            Velocity3::default(),
	            PhysicalEntity::new(Material::new(1.0, 0.05)),
	            Mass3::new(1.0),
            )
            .with(tr)
            .with(
                ForceAccumulator::<Vector3<f32>,Vector3<f32>>::new()
            )
            .build();

        let mut tr = Transform::default();
        // TODO add conf ability to this
        tr.translation = [0.0, 0.35, 0.0].into();
        data.world
            .create_entity()
            .with(tr)
            .with(RotationControl::default())
            .with(Camera::standard_3d(1920.0,1080.0))
            .with(Parent {entity: player_entity})
            .build();

        data.world.add_resource(AmbientColor(Rgba::from([0.1; 3])));
        let mut tr = Transform::default();
        tr.translation = [0.0,10.0,0.0].into();
        tr.rotation = Quaternion::one();
        data.world
        	.create_entity()
        	.with(tr)
        	.with(Light::Point(PointLight {
        		color: Rgba::white(),
        		intensity: 10.0,
        		radius: 60.0,
        		smoothness: 4.0,
        	}))
        	.build();

        let font = data.world.read_resource::<AssetLoader>().load("font/Arial.ttf", FontFormat::Ttf, (), &mut data.world.write_resource(), &mut data.world.write_resource(), &data.world.read_resource()).expect("Failed to load font");
        data.world
            .create_entity()
            .with(UiTransform::new("timer_text".to_string(), Anchor::TopMiddle, 0.0, 500.0, 0.0, 512.0, 256.0, -1))
            .with(UiText::new(font, "timer here".to_string(), [1.0,1.0,1.0,1.0], 30.0))
            .with(Timer::default())
            .build();

        MyPhysicalEntityParts::setup(&mut data.world.res)
    }

    fn update(&mut self, data: &mut StateData<GameData>) -> SimpleTrans<'a, 'b> {
        time_sync(&data.world);

        if !self.init_done && self.load_progress.as_ref().unwrap().is_complete() {
        	info!("BBBBBBBBBBBBBBBBBBBBBBBBBBBBB");
    		let entity_sizes = (&*data.world.entities(), &data.world.read_storage::<Transform>(), &data.world.read_storage::<MeshData>(), &data.world.read_storage::<Named>()).par_join().map(|(entity,transform,mesh_data,name)| {
    			info!("map transform: {:?}", transform);
    			let verts = if let MeshData::Creator(combo) = mesh_data {
    				info!("vertices: {:?}", combo.vertices());
    				combo.vertices().iter().map(|sep| Point3::new((sep.0)[0] * transform.scale.x, (sep.0)[1] * transform.scale.y, (sep.0)[2] * transform.scale.z)).collect::<Vec<_>>()

    			} else {
    				vec![]
    			};
    			(entity,transform.clone(), verts, name.name.clone())
    		}).collect::<Vec<_>>();

    		if !entity_sizes.is_empty() {
    			// The loading is done, now we add the colliders.
    			self.init_done = true;
    			{
    				let mut collider_storage = data.world.write_storage::<ObjectType>();
	    			for (entity, _, _,name) in &entity_sizes {
	    				let (obj_type,coll_strat) = if name == "StartZone" {
	    					(ObjectType::StartZone,CollisionStrategy::CollisionOnly)
	    				} else if name == "EndZone" {
	    					(ObjectType::EndZone,CollisionStrategy::CollisionOnly)
	    				} else if name.starts_with("KillZone") {
	    					(ObjectType::KillZone,CollisionStrategy::CollisionOnly)
	    				} else {
	    					(ObjectType::Scene,CollisionStrategy::FullResolution)
	    				};
	    				collider_storage.insert(*entity, obj_type).expect("Failed to add ObjectType to map mesh");
	    			}
    			}
    			{
	    			let mut physical_parts = MyPhysicalEntityParts::fetch(&mut data.world.res);
	    			for (entity, size, mesh, name) in entity_sizes {
	    				let (obj_type,coll_strat) = if name == "StartZone" {
	    					(ObjectType::StartZone,CollisionStrategy::CollisionOnly)
	    				} else if name == "EndZone" {
	    					(ObjectType::EndZone,CollisionStrategy::CollisionOnly)
	    				} else if name.starts_with("KillZone") {
	    					(ObjectType::KillZone,CollisionStrategy::CollisionOnly)
	    				} else {
	    					(ObjectType::Scene,CollisionStrategy::FullResolution)
	    				};

	    				physical_parts.static_entity(
	    					entity,
	    					Shape::new_simple_with_type(
			                    //CollisionStrategy::FullResolution,
			                    //CollisionStrategy::CollisionOnly,
			                    coll_strat,
			                    CollisionMode::Discrete,
			                    Primitive3::ConvexPolyhedron(<ConvexPolyhedron<f32>>::new(mesh)),
			                    obj_type,
			                ),
			                BodyPose3::new(Point3::new(size.translation.x, size.translation.y,size.translation.z), size.rotation),
			                PhysicalEntity::default(),
			                Mass3::infinite(),
		    			).expect("Failed to add static collider to map mesh");
	    			}
    			}
    		}
        }

        Trans::None
    }
}

fn main() -> amethyst::Result<()> {



	/*


	unique maps to be more than a cs port
	pushed by triggered explosion

	maps totally reset to origin when restarting, thus allowing for cool runtime effects
	possible puzzles?

	hidden doors
	levers activating hidden doors later in the level
	


	*/


    amethyst::start_logger(Default::default());

    let resources_directory = get_working_dir();

    let asset_loader = AssetLoader::new(&format!("{}/assets", resources_directory), "base");

    let display_config_path = asset_loader.resolve_path("config/display.ron").unwrap();
    let display_config = DisplayConfig::load(&display_config_path);

    let key_bindings_path = asset_loader.resolve_path("config/input.ron").unwrap();

    let pipe = Pipeline::build().with_stage(
        Stage::with_backbuffer()
            .clear_target([0.0, 1.0, 0.0, 1.0], 1.0)
            .with_pass(DrawPbmSeparate::new().with_transparency(ColorMask::all(), ALPHA, Some(DepthMode::LessEqualWrite)))
            .with_pass(DrawUi::new())
    );

    let game_data = GameDataBuilder::default()
    	.with(
            PrefabLoaderSystem::<ScenePrefab>::default(),
            "map_loader",
            &[],
        )
        .with(
            PrefabLoaderSystem::<HoppinMapPrefabData>::default(),
            "scene_loader",
            &[],
        )
        .with(
            GltfSceneLoaderSystem::default(),
            "gltf_loader",
            &["scene_loader", "map_loader"],
        )
        .with(FPSRotationRhusicsSystem::<String,String>::new(0.3,0.3), "free_rotation", &[])
        .with(MouseFocusUpdateSystem::new(), "mouse_focus", &[])
        .with(CursorHideSystem::new(), "cursor_hide", &[])
        .with(GroundCheckerSystem, "ground_checker", &[])
        .with(JumpSystem::default(), "jump", &["ground_checker"])
        .with(GroundFrictionSystem, "ground_friction", &["ground_checker", "jump"])
        .with(BhopMovementSystem::<String,String>::new(Some(String::from("right")),Some(String::from("forward"))), "air_bhop_movement", &["free_rotation", "jump", "ground_friction", "ground_checker"])
        .with(GravitySystem, "gravity", &[])
        .with_bundle(TransformBundle::new().with_dep(&[]))?
        .with(ContactSystem::default(), "contacts", &[])
        .with(UiAutoTextSystem::<Timer>::default(), "timer_text_update", &[])
        .with_bundle(
            InputBundle::<String, String>::new().with_bindings_from_file(&key_bindings_path)?,
        )?
        .with_bundle(
            UiBundle::<String, String>::new(),
        )?
        .with_barrier()
        .with_bundle(DefaultPhysicsBundle3::<ObjectType>::new().with_spatial())?
        .with_bundle(RenderBundle::new(pipe, Some(display_config)))?;
    let mut game = Application::build(resources_directory, GameState::default())?
    	.with_resource(asset_loader)
    	.with_resource(AssetLoaderInternal::<FontAsset>::new())
    	.build(game_data)?;
    game.run();
    Ok(())
}
