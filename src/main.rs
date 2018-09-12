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

use amethyst::assets::{PrefabLoader, PrefabLoaderSystem, RonFormat, AssetPrefab, ProgressCounter, PrefabData, AssetStorage, Handle};
use amethyst::assets::Error as AssetError;
use amethyst::ecs::error::Error as ECSError;
use amethyst::shrev::{ReaderId, EventChannel};
use std::hash::Hash;
use amethyst::controls::{FlyControlBundle, FlyControlTag, FlyMovementSystem,WindowFocus,MouseFocusUpdateSystem,FreeRotationSystem,CursorHideSystem};
use amethyst::core::transform::{Transform, TransformBundle, Parent};
use amethyst::core::{WithNamed,Time};
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

use amethyst::core::cgmath::{Matrix3, One, Point3, Quaternion, Vector3, Zero, Deg, Euler, InnerSpace, Rotation, EuclideanSpace, Rotation3, Vector2, SquareMatrix};
use amethyst_rhusics::collision::primitive::{Cuboid, Primitive3, ConvexPolyhedron};
use amethyst_rhusics::collision::{Aabb3,Ray3};
use amethyst_rhusics::rhusics_core::physics3d::{Mass3, Velocity3};
use amethyst_rhusics::rhusics_core::{
    Collider, CollisionMode, CollisionShape, CollisionStrategy, ForceAccumulator, Inertia, Mass,
    Pose, PhysicalEntity, Velocity,Material,NextFrame,
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

#[repr(u8)]
#[derive(Debug, Clone, PartialOrd, PartialEq)]
pub enum ObjectType {
    Scene,
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
        /*self == ObjectType::Player
            && (*other == ObjectType::Scene || *other == ObjectType::KillZone)*/
            //other != self
            true
    }
}

impl Component for ObjectType {
    type Storage = VecStorage<Self>;
}

#[derive(Serialize, Deserialize, Debug, Clone)]
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
        Read<'a, Gravity>,
        WriteStorage<'a, ForceAccumulator<Vector3<f32>, Vector3<f32>>>,
    );
    fn run(&mut self, (gravity, mut forces): Self::SystemData) {
        for (mut force,) in (&mut forces,).join() {
            /*let new_vel = velocity.linear() + gravity.acceleration;
			velocity.set_linear(new_vel);*/
            force.add_force(gravity.acceleration);
        }
    }
}

/// The system that manages the fly movement.
/// Generic parameters are the parameters for the InputHandler.
pub struct FpsMovementSystemSimple<A, B> {
    /// The movement speed of the movement in units per second.
    speed: f32,
    /// The name of the input axis to locally move in the x coordinates.
    right_input_axis: Option<A>,
    /// The name of the input axis to locally move in the y coordinates.
    jump_input_action: Option<B>,
    /// The name of the input axis to locally move in the z coordinates.
    forward_input_axis: Option<A>,
}

impl<A, B> FpsMovementSystemSimple<A, B>
where
    A: Send + Sync + Hash + Eq + Clone + 'static,
    B: Send + Sync + Hash + Eq + Clone + 'static,
{
    pub fn new(
        speed: f32,
        right_input_axis: Option<A>,
        jump_input_action: Option<B>,
        forward_input_axis: Option<A>,
    ) -> Self {
        FpsMovementSystemSimple {
            speed,
            right_input_axis,
            jump_input_action,
            forward_input_axis,
        }
    }

    fn get_axis(name: &Option<A>, input: &InputHandler<A, B>) -> f32 {
        name.as_ref()
            .and_then(|ref n| input.axis_value(n))
            .unwrap_or(0.0) as f32
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
        let x = FpsMovementSystemSimple::get_axis(&self.right_input_axis, &input);
        //let y = FlyMovementSystem::get_axis(&self.up_input_axis, &input);
        let z = -FpsMovementSystemSimple::get_axis(&self.forward_input_axis, &input);

        let dir = Vector3::new(x, 0.0, z);
        if dir.magnitude() != 0.0 {
        	for (transform, _, mut force) in (&transforms, &tags, &mut forces).join() {
	            //transform.move_along_local(dir, time.delta_seconds() * self.speed);
	            info!("Transform data Orientation: {:?}  rotation: {:?}", transform.orientation(), transform.rotation);
	            let mut dir = transform.rotation * dir;
	            dir = dir.normalize();
	            force.add_force(dir * self.speed * time.delta_seconds());
        	}
        }
    }
}

#[derive(Debug,Clone,Default,Serialize,Deserialize)]
pub struct RotationControl{
    pub mouse_accum_x: f32,
    pub mouse_accum_y: f32,
}

impl Component for RotationControl {
	type Storage = DenseVecStorage<Self>;
}

#[derive(Debug,Clone,Default,Serialize,Deserialize)]
pub struct Grounded{
	 pub ground: bool,
	 pub since: f64,
	 pub distance_check: f32,
}

impl Grounded {
	pub fn new(distance_check: f32) -> Self {
		Grounded {
			ground: false,
			since: 0.0,
			distance_check,
		}
	}
}

impl Component for Grounded {
	type Storage = DenseVecStorage<Self>;
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
                            	rotation_control.mouse_accum_y = rotation_control.mouse_accum_y.max(-89.5).min(89.5);
                            	info!("mouseaccum: {:?}", rotation_control);
                            	//transform.rotation = Quaternion::from(Euler::new(Deg(-rotation_control.mouse_accum_y),Deg(0.0),Deg(0.0)));
                            	transform.rotation = Quaternion::from_angle_x(Deg(-rotation_control.mouse_accum_y));

                            	for (mut body_pose, mut next_body_pose, _) in (&mut body_poses, &mut next_body_poses, &fly_controls).join() {
                            		//body_pose.value.set_rotation(Quaternion::from(Euler::new(Deg(0.0),Deg(rotation_control.mouse_accum_x),Deg(0.0))));
                            		body_pose.set_rotation(Quaternion::from_angle_y(Deg(rotation_control.mouse_accum_x)));
                            		next_body_pose.value.set_rotation(Quaternion::from_angle_y(Deg(rotation_control.mouse_accum_x)));
                            	}

                                //transform.pitch_local(Deg(-y as f32 * self.sensitivity_y));
                                //transform.yaw_global(Deg(-x as f32 * self.sensitivity_x));
                                //body_pose.set_rotation(transform.rotation);
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

pub struct GroundChecker;

impl<'a> System<'a> for GroundChecker {
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
			            info!("hit bounding volume of {:?} at point {:?}", v.value, p);
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

#[derive(Default)]
pub struct Jump{
	pub absolute: bool,
	 pub check_ground: bool,
	 pub jump_force: f32,
	 pub auto_jump: bool,
	 pub jump_cooldown: f64,
	 pub input_cooldown: f64,
	 /// Multiplier. Time can go in the negatives.
	 pub jump_timing_boost: Option<PartialFunction<f64,f32>>,
	 pub last_jump: f64,
	 pub last_jump_offset: f64,
}

impl Jump {
	pub fn new(absolute: bool, check_ground: bool, jump_force: f32, auto_jump: bool) -> Self {
		Jump {
			absolute,
			check_ground,
			jump_force,
			auto_jump,
			jump_cooldown: 0.1,
			input_cooldown: 0.1,
			jump_timing_boost: None,
			last_jump: 0.0,
			last_jump_offset: 0.0,
		}
	}
}

impl Component for Jump {
	type Storage = DenseVecStorage<Self>;
}

pub struct JumpSystem{
	last_logical_press: f64,
	/// Was the jump key pressed last frame?
    input_hold: bool,
    /// The last time we physically pressed the jump key.
    last_physical_press: f64,
}

impl JumpSystem {
	pub fn new() -> Self {
		JumpSystem {
			last_logical_press: 0.0,
			input_hold: false,
			last_physical_press: 0.0,
		}
	}
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
		    		info!("Jump!");
		    		if jump.absolute {
		    			force.add_force(Vector3::<f32>::unit_y() * jump.jump_force);
		    		} else {
		    			//velocity.value.linear.y = jump.jump_force;
		    			velocity.value.set_linear(Vector3::new(velocity.value.linear().x, jump.jump_force, velocity.value.linear().z));
		    		}
		    		// TODO: jump boost curve

	    		}
	    	}
        } else {
        	// The jump key was released.
        	self.input_hold = false;
        }
    }
}

#[derive(Deserialize, Serialize)]
struct HoppinMapPrefabData {
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












pub struct GroundMovement3D {
	/// False = Forces, True = Velocity
	pub absolute: bool,
    /// Use world coordinates XYZ.
	pub absolute_axis: bool,
    /// Negates the velocity when pressing the key opposite to the current velocity.
	pub counter_impulse: bool,
    /// Acceleration in unit/s²
    pub accelerate: f32,
    /// Acceleration when sprinting in unit/s² if sprinting is enabled
    pub accelerate_sprint: f32,
    /// The maximum ground velocity.
    pub max_velocity: f32,
}

impl GroundMovement3D {
	pub fn new(absolute: bool, accelerate: f32, accelerate_sprint: Option<f32>, max_velocity: f32) -> Self {
		let accelerate_sprint = accelerate_sprint.unwrap_or(0.0);
		GroundMovement3D {
			absolute,
			absolute_axis: false,
			counter_impulse: false,
			accelerate,
			accelerate_sprint,
			max_velocity,
		}
	}
}

impl Component for GroundMovement3D {
	type Storage = VecStorage<Self>;
}

pub struct AirMovement3D {
	/// False = Forces, True = Velocity
	pub absolute: bool,
    /// Use world coordinates XYZ.
	pub absolute_axis: bool,
    /// Negates the velocity when pressing the key opposite to the current velocity.
	pub counter_impulse: bool,
	/// Enables accelerating over maximumVelocity by airstrafing. Bunnyhop in a nutshell.
	pub allow_projection_acceleration: bool,
    /// Acceleration in unit/s²
    pub accelerate: f32,
    /// The maximum ground velocity.
    pub max_velocity: f32,
}

impl AirMovement3D {
	pub fn new(absolute: bool, allow_projection_acceleration: bool, accelerate: f32, max_velocity: f32) -> Self {
		AirMovement3D {
			absolute,
			absolute_axis: false,
			counter_impulse: false,
			allow_projection_acceleration,
			accelerate,
			max_velocity,
		}
	}
}

impl Component for AirMovement3D {
	type Storage = VecStorage<Self>;
}


pub enum FrictionMode {
	Linear,
	Percent,
}

pub struct GroundFriction3D {
	/// The amount of friction speed loss by second.
	pub friction: f32,
	/// The way friction is applied.
	pub friction_mode: FrictionMode,
	/// The time to wait after touching the ground before applying the friction.
	pub ground_time_before_apply: f64,
}

impl GroundFriction3D {
	pub fn new(friction: f32, friction_mode: FrictionMode, ground_time_before_apply: f64) -> Self {
		GroundFriction3D {
			friction,
			friction_mode,
			ground_time_before_apply,
		}
	}
}

impl Component for GroundFriction3D {
	type Storage = VecStorage<Self>;
}


/// The system that manages the fly movement.
/// Generic parameters are the parameters for the InputHandler.
pub struct GroundBhopMovementSystem<A, B> {
    /// The name of the input axis to locally move in the x coordinates.
    right_input_axis: Option<A>,
    /// The name of the input axis to locally move in the z coordinates.
    forward_input_axis: Option<A>,
    phantom_data: PhantomData<B>,
}

impl<A, B> GroundBhopMovementSystem<A, B>
where
    A: Send + Sync + Hash + Eq + Clone + 'static,
    B: Send + Sync + Hash + Eq + Clone + 'static,
{
    pub fn new(
        right_input_axis: Option<A>,
        forward_input_axis: Option<A>,
    ) -> Self {
        GroundBhopMovementSystem {
            right_input_axis,
            forward_input_axis,
            phantom_data: PhantomData,
        }
    }

    fn get_axis(name: &Option<A>, input: &InputHandler<A, B>) -> f32 {
        name.as_ref()
            .and_then(|ref n| input.axis_value(n))
            .unwrap_or(0.0) as f32
    }
}

impl<'a, A, B> System<'a> for GroundBhopMovementSystem<A, B>
where
    A: Send + Sync + Hash + Eq + Clone + 'static,
    B: Send + Sync + Hash + Eq + Clone + 'static,
{
    type SystemData = (
        Read<'a, Time>,
        Read<'a, InputHandler<A, B>>,
        ReadStorage<'a, GroundMovement3D>,
        WriteStorage<'a, ForceAccumulator<Vector3<f32>, Vector3<f32>>>,
        WriteStorage<'a, NextFrame<Velocity3<f32>>>,
    );

    fn run(&mut self, (time, input, ground_movements, mut forces, mut velocities): Self::SystemData) {
        /*let x = GroundBhopMovementSystem::get_axis(&self.right_input_axis, &input);
        //let y = FlyMovementSystem::get_axis(&self.up_input_axis, &input);
        let z = -GroundBhopMovementSystem::get_axis(&self.forward_input_axis, &input);

        let dir = Vector3::new(x, 0.0, z);
        if dir.magnitude() != 0.0 {
        	for (transform, _, mut force) in (&transforms, &tags, &mut forces).join() {
	            //transform.move_along_local(dir, time.delta_seconds() * self.speed);
	            info!("Transform data Orientation: {:?}  rotation: {:?}", transform.orientation(), transform.rotation);
	            let mut dir = transform.rotation * dir;
	            dir = dir.normalize();
	            force.add_force(dir * self.speed * time.delta_seconds());
        	}
        }*/
    }
}



/// The system that manages the fly movement.
/// Generic parameters are the parameters for the InputHandler.
pub struct AirBhopMovementSystem<A, B> {
    /// The name of the input axis to locally move in the x coordinates.
    right_input_axis: Option<A>,
    /// The name of the input axis to locally move in the z coordinates.
    forward_input_axis: Option<A>,
    phantom_data: PhantomData<B>,
}

impl<A, B> AirBhopMovementSystem<A, B>
where
    A: Send + Sync + Hash + Eq + Clone + 'static,
    B: Send + Sync + Hash + Eq + Clone + 'static,
{
    pub fn new(
        right_input_axis: Option<A>,
        forward_input_axis: Option<A>,
    ) -> Self {
        AirBhopMovementSystem {
            right_input_axis,
            forward_input_axis,
            phantom_data: PhantomData,
        }
    }

    fn get_axis(name: &Option<A>, input: &InputHandler<A, B>) -> f32 {
        name.as_ref()
            .and_then(|ref n| input.axis_value(n))
            .unwrap_or(0.0) as f32
    }
}

impl<'a, A, B> System<'a> for AirBhopMovementSystem<A, B>
where
    A: Send + Sync + Hash + Eq + Clone + 'static,
    B: Send + Sync + Hash + Eq + Clone + 'static,
{
    type SystemData = (
        Read<'a, Time>,
        Read<'a, InputHandler<A, B>>,
        ReadStorage<'a, Transform>,
        ReadStorage<'a, AirMovement3D>,
        ReadStorage<'a, Grounded>,
        WriteStorage<'a, ForceAccumulator<Vector3<f32>, Vector3<f32>>>,
        WriteStorage<'a, NextFrame<Velocity3<f32>>>,
    );

    fn run(&mut self, (time, input, transforms, air_movements, groundeds, mut forces, mut velocities): Self::SystemData) {
    	let x = AirBhopMovementSystem::get_axis(&self.right_input_axis, &input);
	    let z = -AirBhopMovementSystem::get_axis(&self.forward_input_axis, &input);
	    let input = Vector2::new(x, z);
	    let input3 = Vector3::new(x, 0.0, z);

	    if input3.magnitude() != 0.0 {
	    	for (transform, air_movement, grounded, mut force, mut velocity) in (&transforms, &air_movements, &groundeds, &mut forces, &mut velocities).join() {
	    		if grounded.ground {
	    			continue;
	    		}

	            let mut relative = transform.rotation * input3;
	            relative = relative.normalize();

	            if air_movement.absolute {
	            	// multiply by transform.rotation.inverse()
	            	// movementBase.RigidBody ().velocity = transform.TransformDirection (new Vector3 (input.x, rel.y, input.y));
	            	// https://www.amethyst.rs/doc/develop/doc/src/amethyst_core/transform/components/local_transform.rs.html#300
	            	let new_vel = transform.matrix().invert().unwrap() * Vector3::new(input.x, relative.y, input.y);
	            	velocity.value.set_linear(new_vel);
	            } else {
	            	let mut wish_vel = relative;
	            	if air_movement.counter_impulse {
	            		wish_vel = counter_impulse(input, wish_vel);
	            	}

	            	wish_vel = accelerate_vector(time.delta_seconds(), input, relative, air_movement.accelerate, air_movement.max_velocity);
	            	if !air_movement.allow_projection_acceleration {
	            		wish_vel = limit_velocity(wish_vel, air_movement.max_velocity);
	            	}

	            	//movementBase.RigidBody().velocity = transform.TransformDirection(wishVel);
	            }
	    	}
    	}
    }
}




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
	    		match friction.friction_mode {
	    			FrictionMode::Linear => {
						let slowdown = friction.friction * time.delta_seconds();
		                velocity.value.set_linear(Vector3::new(apply_friction_single(velocity.value.linear().x, slowdown),velocity.value.linear().y,apply_friction_single(velocity.value.linear().z, slowdown)));
	    			},
	    			FrictionMode::Percent => {
						let vel = velocity.value.linear();
		                let coef = friction.friction * time.delta_seconds();
		                velocity.value.set_linear(Vector3::new(apply_friction_single(vel.x, vel.x*coef),vel.y,apply_friction_single(vel.z, vel.z*coef)));
	    			},
	    		}
    		}
    	}
        /*let dir = Vector3::new(x, 0.0, z);
        if dir.magnitude() != 0.0 {
        	for (transform, _, mut force) in (&transforms, &tags, &mut forces).join() {
	            //transform.move_along_local(dir, time.delta_seconds() * self.speed);
	            info!("Transform data Orientation: {:?}  rotation: {:?}", transform.orientation(), transform.rotation);
	            let mut dir = transform.rotation * dir;
	            dir = dir.normalize();
	            force.add_force(dir * self.speed * time.delta_seconds());
        	}
        }*/
    }
}


pub fn accelerate_vector(delta_time: f32, input: Vector2<f32>, rel: Vector3<f32>, force: f32, max_velocity: f32) -> Vector3<f32> {
	let mut o = rel;
    let input3 = Vector3::new(input.x, 0.0, input.y);
    let rel_flat = Vector3::new(rel.x, 0.0, rel.z);
    if input3.magnitude() > 0.0 {
        let proj = rel_flat.dot(input3.normalize());
        let mut accel_velocity = force * delta_time as f32;
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


pub fn limit_velocity(vec: Vector3<f32>, maximum_velocity: f32) -> Vector3<f32> {
    let v = Vector2::new(vec.x,vec.z).magnitude();
    if v > maximum_velocity && maximum_velocity != 0.0 {
        let ratio = maximum_velocity / v;
        info!("LIMITING RATIO={}",ratio);
        return Vector3::new(vec.x * ratio,vec.y,vec.z * ratio);
    }
    vec
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

        //data.world.create_entity().with(prefab_handle).build();
        data.world.create_entity().with(scene_handle).build();

        data.world.register::<ObjectType>();

        let mut tr = Transform::default();
        //tr.translation = [-8.0,5.0,-36.0].into();
        tr.translation = [0.0,5.0,0.0].into();

        let mut ground_movement = GroundMovement3D::new(false, 50.0, None, 5.5);
        let mut air_movement = AirMovement3D::new(false, true, 10000.0, 2.0);
        let mut ground_friction = GroundFriction3D::new(2.0, FrictionMode::Percent, 0.2);

        let player_entity = data.world
            .create_entity()
            .with(FlyControlTag)
            .with(ObjectType::Player)
            .with(Grounded::new(0.5))
            .with(ground_movement)
            .with(air_movement)
            .with(ground_friction)
            .with(Jump::new(true, true, 20.0, true))
            .with_dynamic_physical_entity(
            	CollisionShape::<Primitive3<f32>, BodyPose3<f32>, Aabb3<f32>, ObjectType>::new_simple(
	                CollisionStrategy::FullResolution,
	                //CollisionMode::Continuous,
	                CollisionMode::Discrete,
	                Cuboid::new(0.1, 1.0, 0.1).into(),
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

        let camera_entity = data.world
            .create_entity()
            .with(Transform::default())
            .with(RotationControl::default())
            .with(Camera::standard_3d(1920.0,1080.0))
            .with(Parent {entity: player_entity})
            .build();

        // floor collider temp
        /*let mut tr = Transform::default();
        tr.translation = [0.0,-5.0,0.0].into();
        data.world
            .create_entity()
            .with(ObjectType::Scene)
            .with_static_physical_entity(
                Shape::new_simple_with_type(
                    CollisionStrategy::FullResolution,
                    //CollisionMode::Continuous,
                    CollisionMode::Discrete,
                    Cuboid::new(5.0, 1.0,5.0).into(),
                    ObjectType::Scene,
                ),
                BodyPose3::new(Point3::new(tr.translation.x, tr.translation.y,tr.translation.z), Quaternion::one()),
                PhysicalEntity::default(),
                Mass3::infinite(),
            )
            .with(tr)
            .build();*/

        data.world.add_resource(AmbientColor(Rgba::from([0.1; 3])));
        /*let mut rbp = RigidBodyParts::new(&data.world);
        rbp.dynamic_body(
        	player_entity,
        	            CollisionShape::<Primitive3<f32>, BodyPose3<f32>, Aabb3<f32>, ObjectType>::new_simple(
                CollisionStrategy::FullResolution,
                CollisionMode::Continuous,
                Cuboid::new(0.5, 1.0, 0.5).into(),
            ),
            BodyPose3::new(Point3::new(0.0, 0.0, 0.0), Quaternion::<f32>::one()),
            Velocity3::default(),
            RigidBody::<f32>::default(),
            Mass3::new(1.0),
        ).unwrap();*/
        PhysicalEntityParts::<Primitive3<f32>,ObjectType,Quaternion<f32>,Vector3<f32>,Vector3<f32>,Matrix3<f32>,Aabb3<f32>,BodyPose3<f32>>::setup(&mut data.world.res)
    }

    fn update(&mut self, data: &mut StateData<GameData>) -> SimpleTrans<'a, 'b> {
        time_sync(&data.world);

        if !self.init_done && self.load_progress.as_ref().unwrap().is_complete() {
        	//self.init_done = true;

    		/*let entity_sizes = (&*data.world.entities(), &data.world.read_storage::<MeshData>(),).par_join().map(|mesh_data| {
    			/*let mesh = mesh_storage.get(mesh_handle.0).expect("Failed to find mesh for mesh handle.");
    			let raw_buffer = &*mesh.buffer(&[("position",<PosNormTex as With<Position>>::FORMAT)]).expect("Failed to get mesh data from gpu");*/
    			()
    		}).collect::<Vec<_>>();*/


    		let entity_sizes = (&*data.world.entities(), &data.world.read_storage::<Transform>(), &data.world.read_storage::<MeshData>()).par_join().map(|(entity,transform,mesh_data)| {
    			info!("map transform: {:?}", transform);
    			let verts = if let MeshData::Creator(combo) = mesh_data {
    				info!("vertices: {:?}", combo.vertices());
    				combo.vertices().iter().map(|sep| Point3::new((sep.0)[0] * transform.scale.x, (sep.0)[1] * transform.scale.y, (sep.0)[2] * transform.scale.z)).collect::<Vec<_>>()

    			} else {
    				vec![]
    			};
    			(entity,transform.clone(), verts)
    		}).collect::<Vec<_>>();

    		if !entity_sizes.is_empty() {
    			// The loading is done, now we add the colliders.
    			self.init_done = true;
    			{
    				let mut collider_storage = data.world.write_storage::<ObjectType>();
	    			for (entity, _, _) in &entity_sizes {
	    				collider_storage.insert(*entity, ObjectType::Scene).expect("Failed to add ObjectType to map mesh");
	    			}
    			}
    			{
	    			let mut physical_parts = PhysicalEntityParts::<Primitive3<f32>,ObjectType,Quaternion<f32>,Vector3<f32>,Vector3<f32>,Matrix3<f32>,Aabb3<f32>,BodyPose3<f32>>::fetch(&mut data.world.res);
	    			for (entity, size, mesh) in entity_sizes {
	    				info!("\nADDING!!!!!!!!!!\nTransform: {:?}\nMesh:{:?}", size, mesh);
	    				physical_parts.static_entity(
	    					entity,
	    					Shape::new_simple_with_type(
			                    CollisionStrategy::FullResolution,
			                    CollisionMode::Discrete,
			                    //Cuboid::new(size.scale.x*2.0, size.scale.y*2.0, size.scale.z*2.0).into(),
			                    Primitive3::ConvexPolyhedron(<ConvexPolyhedron<f32>>::new(mesh)),
			                    ObjectType::Scene,
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
            //.with_pass(DrawPbmSeparate::new())
            .with_pass(DrawPbmSeparate::new().with_transparency(ColorMask::all(), ALPHA, Some(DepthMode::LessEqualWrite)))
            //.with_pass(DrawUi::new()),
    );

    let game_data = GameDataBuilder::default()
        .with(FPSRotationRhusicsSystem::<String,String>::new(0.3,0.3), "free_rotation", &[])
        .with(MouseFocusUpdateSystem::new(), "mouse_focus", &[])
        .with(CursorHideSystem::new(), "cursor_hide", &[])
        .with(GroundChecker, "ground_checker", &[])
        .with(JumpSystem::new(), "jump", &["ground_checker"])
        .with(BhopMovementSystem::new(Some(String::from("right")),Some(String::from("forward"))),"bhop_movement", &["free_rotation", "jump"])
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
        	"map_loader2",
        	&[],
        )
        .with(GravitySystem, "gravity", &[])
        .with(
            GltfSceneLoaderSystem::default(),
            "gltf_loader",
            &[],
        )
        .with_bundle(TransformBundle::new().with_dep(&[]))?
        .with_bundle(
            InputBundle::<String, String>::new().with_bindings_from_file(&key_bindings_path)?,
        )?
        .with_barrier()
        .with_bundle(DefaultPhysicsBundle3::<ObjectType>::new().with_spatial())?
        .with_bundle(RenderBundle::new(pipe, Some(display_config)))?;
    let mut game = Application::build(resources_directory, GameState::default())?.build(game_data)?;
    game.run();
    Ok(())
}
