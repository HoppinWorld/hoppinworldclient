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

use amethyst::assets::{PrefabLoader, PrefabLoaderSystem, RonFormat, AssetPrefab, ProgressCounter, PrefabData};
use amethyst::assets::Error as AssetError;
use amethyst::ecs::error::Error as ECSError;
use amethyst::shrev::{ReaderId, EventChannel};
use std::hash::Hash;
use amethyst::controls::{FlyControlBundle, FlyControlTag, FlyMovementSystem,WindowFocus,MouseFocusUpdateSystem,FreeRotationSystem,CursorHideSystem};
use amethyst::core::transform::{Transform, TransformBundle, Parent};
use amethyst::core::{WithNamed,Time};
use amethyst::ecs::{
    Component, DenseVecStorage, Join, Read, ReadStorage, System, VecStorage, Write, WriteStorage, Resources, SystemData, Entity, Entities,
};
use amethyst::input::{InputBundle, InputHandler};
use amethyst::prelude::*;
use amethyst::renderer::*;
use amethyst::utils::scene::BasicScenePrefab;
use amethyst::Error;
use amethyst_gltf::{GltfSceneAsset, GltfSceneFormat, GltfSceneLoaderSystem};

use amethyst::core::cgmath::{Matrix3, One, Point3, Quaternion, Vector3, Zero, Deg, Euler, InnerSpace, Rotation, EuclideanSpace};
use amethyst_rhusics::collision::primitive::{Cuboid, Primitive3};
use amethyst_rhusics::collision::{Aabb3,Ray3};
use amethyst_rhusics::rhusics_core::physics3d::{Mass3, Velocity3};
use amethyst_rhusics::rhusics_core::{
    Collider, CollisionMode, CollisionShape, CollisionStrategy, ForceAccumulator, Inertia, Mass,
    Pose, PhysicalEntity, Velocity,Material,NextFrame,
};
use amethyst_rhusics::rhusics_ecs::physics3d::{BodyPose3,DynamicBoundingVolumeTree3};
use amethyst_rhusics::collision::dbvt::query_ray;
use amethyst_rhusics::rhusics_ecs::WithPhysics;
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
pub struct FpsMovementSystem<A, B> {
    /// The movement speed of the movement in units per second.
    speed: f32,
    /// The name of the input axis to locally move in the x coordinates.
    right_input_axis: Option<A>,
    /// The name of the input axis to locally move in the y coordinates.
    jump_input_action: Option<B>,
    /// The name of the input axis to locally move in the z coordinates.
    forward_input_axis: Option<A>,
}

impl<A, B> FpsMovementSystem<A, B>
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
        FpsMovementSystem {
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

impl<'a, A, B> System<'a> for FpsMovementSystem<A, B>
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
        let x = FpsMovementSystem::get_axis(&self.right_input_axis, &input);
        //let y = FlyMovementSystem::get_axis(&self.up_input_axis, &input);
        let z = -FpsMovementSystem::get_axis(&self.forward_input_axis, &input);

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
        WriteStorage<'a, NextFrame<BodyPose3<f32>>>,
        WriteStorage<'a, RotationControl>,
        ReadStorage<'a, FlyControlTag>,
        Read<'a, WindowFocus>,
    );

    fn run(&mut self, (events, mut transforms, mut body_poses, mut rotation_controls, fly_controls,focus): Self::SystemData) {
        let focused = focus.is_focused;
        for event in events.read(&mut self.event_reader.as_mut().unwrap()) {
            if focused {
                match *event {
                    Event::DeviceEvent { ref event, .. } => match *event {
                        DeviceEvent::MouseMotion { delta: (x, y) } => {
                            for (mut transform, mut rotation_control) in (&mut transforms, &mut rotation_controls).join() {
                            	rotation_control.mouse_accum_x -= x as f32 * self.sensitivity_x;

                            	rotation_control.mouse_accum_y += y as f32 * self.sensitivity_y;
                            	rotation_control.mouse_accum_y = rotation_control.mouse_accum_y.max(-89.99999).min(89.99999);
                            	info!("mouseaccum: {:?}", rotation_control);
                            	transform.rotation = Quaternion::from(Euler::new(Deg(-rotation_control.mouse_accum_y),Deg(0.0),Deg(0.0)));

                            	for (mut body_pose, _) in (&mut body_poses, &fly_controls).join() {
                            		body_pose.value.set_rotation(Quaternion::from(Euler::new(Deg(0.0),Deg(rotation_control.mouse_accum_x),Deg(0.0))));
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
	pub fn new(check_ground: bool, jump_force: f32, auto_jump: bool) -> Self {
		Jump {
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
    );

    fn run(&mut self, (entities, grounded, mut jumps, time, input, mut forces): Self::SystemData) {
    	if let Some(true) = input.action_is_down("jump") {
    		if !self.input_hold {
    			// We just started pressing the key. Registering time.
    			self.last_physical_press = time.absolute_time_seconds();
    			self.input_hold = true;
    		}

	    	for (entity, mut jump, mut force) in (&*entities, &mut jumps, &mut forces).join() {
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
		    		force.add_force(Vector3::<f32>::unit_y() * jump.jump_force);
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

struct GameState;

impl<'a, 'b> SimpleState<'a, 'b> for GameState {
    fn on_start(&mut self, data: StateData<GameData>) {
        /*let prefab_handle = data.world.exec(|loader: PrefabLoader<ScenePrefab>| {
            loader.load("assets/base/maps/test.ron", RonFormat, (), ())
        });*/

        let scene_handle = data.world.exec(
            |loader: PrefabLoader<HoppinMapPrefabData>| {
                loader.load(
                    "assets/base/maps/test01.hop",
                    //"assets/base/tmp/test.ron",
                    RonFormat,
                    (),
                    (),
                )
            }
        );

        //data.world.create_entity().with(prefab_handle).build();
        data.world.create_entity().with(scene_handle).build();

        data.world.register::<ObjectType>();

        let player_entity = data.world
            .create_entity()
            .with(Transform::default())
            .with(FlyControlTag)
            .with(ObjectType::Player)
            .with(Grounded::new(0.5))
            .with(Jump::new(true, 20.0, true))
            .with_dynamic_physical_entity(
            	CollisionShape::<Primitive3<f32>, BodyPose3<f32>, Aabb3<f32>, ObjectType>::new_simple(
	                CollisionStrategy::FullResolution,
	                CollisionMode::Continuous,
	                Cuboid::new(0.5, 1.0, 0.5).into(),
	            ),
	            BodyPose3::new(Point3::new(0.0, 0.0, 0.0), Quaternion::<f32>::one()),
	            Velocity3::default(),
	            PhysicalEntity::new(Material::new(1.0, 0.05)),
	            Mass3::new(1.0),
            )
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
        let mut tr = Transform::default();
        tr.translation = [0.0,-5.0,0.0].into();
        data.world
            .create_entity()
            .with(ObjectType::Scene)
            .with_static_physical_entity(
                Shape::new_simple_with_type(
                    CollisionStrategy::FullResolution,
                    CollisionMode::Continuous,
                    Cuboid::new(5.0, 1.0,5.0).into(),
                    ObjectType::Scene,
                ),
                BodyPose3::new(Point3::new(tr.translation.x, tr.translation.y,tr.translation.z), Quaternion::one()),
                PhysicalEntity::default(),
                Mass3::infinite(),
            )
            .with(tr)
            .build();

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
    }

    fn update(&mut self, data: &mut StateData<GameData>) -> SimpleTrans<'a, 'b> {
        time_sync(&data.world);
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
        .with(FpsMovementSystem::new(50.0,Some(String::from("right")),Some(String::from("jump")),Some(String::from("forward"))),"fps_movement", &["free_rotation"])
        .with(JumpSystem::new(), "jump", &["ground_checker"])
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
    let mut game = Application::build(resources_directory, GameState)?.build(game_data)?;
    game.run();
    Ok(())
}
