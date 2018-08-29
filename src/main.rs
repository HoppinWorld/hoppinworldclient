extern crate amethyst;
extern crate amethyst_extra;
extern crate amethyst_rhusics;
#[macro_use]
extern crate serde;
#[macro_use]
extern crate log;

use amethyst::assets::{PrefabLoader, PrefabLoaderSystem, RonFormat};
use std::hash::Hash;
use amethyst::controls::{FlyControlBundle, FlyControlTag, FlyMovementSystem};
use amethyst::core::transform::{Transform, TransformBundle};
use amethyst::core::{WithNamed,Time};
use amethyst::ecs::{
    Component, DenseVecStorage, Join, Read, ReadStorage, System, VecStorage, Write, WriteStorage,
};
use amethyst::input::{InputBundle, InputHandler};
use amethyst::prelude::*;
use amethyst::renderer::{
    Camera, DisplayConfig, DrawShaded, Pipeline, PosNormTex, RenderBundle, Stage,
};
use amethyst::utils::scene::BasicScenePrefab;
use amethyst::Error;

use amethyst::core::cgmath::{Matrix3, One, Point3, Quaternion, Vector3, Zero};
use amethyst_rhusics::collision::primitive::{Cuboid, Primitive3};
use amethyst_rhusics::collision::Aabb3;
use amethyst_rhusics::rhusics_core::physics3d::{Mass3, Velocity3};
use amethyst_rhusics::rhusics_core::{
    Collider, CollisionMode, CollisionShape, CollisionStrategy, ForceAccumulator, Inertia, Mass,
    Pose, RigidBody, Velocity,
};
use amethyst_rhusics::rhusics_ecs::physics3d::BodyPose3;
use amethyst_rhusics::rhusics_ecs::WithRigidBody;
use amethyst_rhusics::{time_sync, DefaultPhysicsBundle3};

use amethyst_extra::*;

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
        /**self == ObjectType::Player
            && (*other == ObjectType::Scene || *other == ObjectType::KillZone)*/
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

    fn run(&mut self, (time, mut transforms, input, tags, mut forces): Self::SystemData) {
        let x = FpsMovementSystem::get_axis(&self.right_input_axis, &input);
        //let y = FlyMovementSystem::get_axis(&self.up_input_axis, &input);
        let z = FpsMovementSystem::get_axis(&self.forward_input_axis, &input);

        let dir = Vector3::new(x, 0.0, z);

        for (_, _, mut force) in (&transforms, &tags, &mut forces).join() {
            //transform.move_along_local(dir, time.delta_seconds() * self.speed);
            force.add_force(dir * self.speed);
        }
    }
}

struct GameState;

impl<'a, 'b> SimpleState<'a, 'b> for GameState {
    fn on_start(&mut self, data: StateData<GameData>) {
        let prefab_handle = data.world.exec(|loader: PrefabLoader<ScenePrefab>| {
            loader.load("assets/base/maps/test.ron", RonFormat, (), ())
        });
        data.world.create_entity().with(prefab_handle).build();

        data.world.register::<ObjectType>();

        let player_entity = data.world
            .create_entity()
            .with(Transform::default())
            .with(FlyControlTag)
            .with(Camera::standard_3d(1920.0,1080.0))
            .with(ObjectType::Player)
            .with_dynamic_rigid_body(
            	CollisionShape::<Primitive3<f32>, BodyPose3<f32>, Aabb3<f32>, ObjectType>::new_simple(
	                CollisionStrategy::FullResolution,
	                CollisionMode::Continuous,
	                Cuboid::new(0.5, 1.0, 0.5).into(),
	            ),
	            BodyPose3::new(Point3::new(0.0, 0.0, 0.0), Quaternion::<f32>::one()),
	            Velocity3::default(),
	            RigidBody::<f32>::default(),
	            Mass3::new(1.0),
            )
            .with(
                ForceAccumulator::<Vector3<f32>,Vector3<f32>>::new()
            )
            .build();

        // floor collider temp
        let mut tr = Transform::default();
        tr.translation = [0.0,-5.0,0.0].into();
        data.world
            .create_entity()
            .with(ObjectType::Scene)
            .with_static_rigid_body(
                Shape::new_simple_with_type(
                    CollisionStrategy::FullResolution,
                    CollisionMode::Discrete,
                    Cuboid::new(5.0, 1.0,5.0).into(),
                    ObjectType::Scene,
                ),
                BodyPose3::new(Point3::new(tr.translation.x, tr.translation.y,tr.translation.z), Quaternion::one()),
                RigidBody::default(),
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
    amethyst::start_logger(Default::default());

    let resources_directory = get_working_dir();

    let asset_loader = AssetLoader::new(&format!("{}/assets", resources_directory), "base");

    let display_config_path = asset_loader.resolve_path("config/display.ron").unwrap();
    let display_config = DisplayConfig::load(&display_config_path);

    let key_bindings_path = asset_loader.resolve_path("config/input.ron").unwrap();

    let pipe = Pipeline::build().with_stage(
        Stage::with_backbuffer()
            .clear_target([0.0, 1.0, 0.0, 1.0], 1.0)
            .with_pass(DrawShaded::<PosNormTex>::new())
            //.with_pass(DrawUi::new()),
    );

    let game_data = GameDataBuilder::default()
        .with_bundle(
            FlyControlBundle::<String, String>::new(
                Some(String::from("right")),
                Some(String::from("jump")),
                Some(String::from("forward")),
            ).with_sensitivity(0.1, 0.1),
        )?
        .with(
            PrefabLoaderSystem::<ScenePrefab>::default(),
            "map_loader",
            &[],
        )
        .with(GravitySystem, "gravity", &[])
        .with_bundle(TransformBundle::new().with_dep(&["fly_movement"]))?
        .with_bundle(
            InputBundle::<String, String>::new().with_bindings_from_file(&key_bindings_path)?,
        )?
        .with_bundle(DefaultPhysicsBundle3::<ObjectType>::new().with_spatial())?
        .with_bundle(RenderBundle::new(pipe, Some(display_config)))?;
    let mut game = Application::build(resources_directory, GameState)?.build(game_data)?;
    game.run();
    Ok(())
}
