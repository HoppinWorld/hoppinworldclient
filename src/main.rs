#[macro_use]
extern crate amethyst;
//extern crate amethyst_core;
extern crate amethyst_extra;
extern crate amethyst_gltf;
extern crate amethyst_rhusics;
#[macro_use]
extern crate serde;
#[macro_use]
extern crate serde_json;
#[macro_use]
extern crate log;
extern crate partial_function;
extern crate winit;
#[macro_use]
extern crate derive_new;
#[macro_use]
extern crate specs_derive;
extern crate ron;
extern crate uuid;
extern crate hoppinworlddata;
extern crate hoppinworldruntime;
//extern crate amethyst_editor_sync;
extern crate hyper;
extern crate hyper_tls;
extern crate tokio;
extern crate tokio_executor;


/*#[macro_use]
extern crate derive_builder;*/

use hoppinworldruntime::*;
use std::sync::{Arc, Mutex};
//use amethyst_editor_sync::*;
use amethyst::utils::removal::Removal;
use std::collections::VecDeque;
use amethyst::assets::Prefab;
use amethyst::assets::{
    PrefabLoaderSystem
};
use amethyst::controls::{
    CursorHideSystem, MouseFocusUpdateSystem
};
use amethyst::core::transform::TransformBundle;
use amethyst::core::Time;
use amethyst::ecs::{
    Entities, Entity, Join, Read, ReadStorage, Resources,
    System, SystemData, Write, WriteStorage,
};
use amethyst::input::InputBundle;
use amethyst::prelude::*;
use amethyst::renderer::{MeshData, ALPHA, ColorMask,
    DepthMode, RenderBundle, DrawPbmSeparate, Pipeline, DisplayConfig, Stage};
use amethyst::shrev::{EventChannel, ReaderId};
use amethyst::ui::*;
use amethyst_gltf::*;


use amethyst::utils::fps_counter::FPSCounterBundle;
use amethyst::core::cgmath::{
    Point3, Vector3,
};
use amethyst_extra::*;
use amethyst_rhusics::rhusics_core::physics3d::Velocity3;
use amethyst_rhusics::rhusics_core::{
    ContactEvent, NextFrame
};
use amethyst_rhusics::DefaultPhysicsBundle3;
use hyper::{Body, Client, Request};
use hyper_tls::HttpsConnector;
use tokio::prelude::{Future, Stream};
use tokio::runtime::Runtime;


pub mod component;
pub mod resource;
pub mod state;
pub mod system;
pub mod util;

use self::component::*;
use self::resource::*;
use self::state::*;
use self::system::*;
use self::util::*;


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

#[derive(Serialize, Deserialize, Clone)]
pub struct Auth {
    token: String,
}

/// Calculates in relative time using the internal engine clock.
#[derive(Default, Serialize)]
pub struct RelativeTimer {
    pub start: f64,
    pub current: f64,
    pub running: bool,
}

impl RelativeTimer {
    pub fn get_text(&self) -> String {
        sec_to_display(self.duration())
    }
    pub fn duration(&self) -> f64 {
        self.current - self.start
    }
    pub fn start(&mut self, cur_time: f64) {
        self.start = cur_time;
        self.current = cur_time;
        self.running = true;
    }
    pub fn update(&mut self, cur_time: f64) {
        if self.running {
            self.current = cur_time;
        }
    }
    pub fn stop(&mut self) {
        self.running = false;
    }
}


pub fn sec_to_display(secs: f64) -> String {
    if secs > -0.00001 && secs < 0.00001 {
        String::from("-")
    } else {
        ((secs * 1000.0).ceil() / 1000.0).to_string()
    }
}

pub struct RelativeTimerSystem;

impl<'a> System<'a> for RelativeTimerSystem {
    type SystemData = (Write<'a, RelativeTimer>, Read<'a, Time>);
    fn run(&mut self, (mut timer, time): Self::SystemData) {
        timer.update(time.absolute_time_seconds());
    }
}


#[derive(Default)]
pub struct ColliderGroundedSystem {
    contact_reader: Option<ReaderId<ContactEvent<Entity, Point3<f32>>>>,
}

impl<'a> System<'a> for ColliderGroundedSystem {
    type SystemData = (
        Entities<'a>,
        Read<'a, EventChannel<ContactEvent<Entity, Point3<f32>>>>,
        Read<'a, Time>,
        ReadStorage<'a, ObjectType>,
        ReadStorage<'a, PlayerTag>,
        WriteStorage<'a, Grounded>,
    );

    fn run(
        &mut self,
        (
            entities,
            contacts,
            time,
            object_types,
            players,
            mut groundeds,
        ): Self::SystemData,
    ) {
        let mut ground = false;
        for contact in contacts.read(&mut self.contact_reader.as_mut().unwrap()) {
            //info!("Collision: {:?}",contact);
            let type1 = object_types.get(contact.bodies.0);
            let type2 = object_types.get(contact.bodies.1);

            if type1.is_none() || type2.is_none() {
                continue;
            }
            let type1 = type1.unwrap();
            let type2 = type2.unwrap();

            //info!("CONTACT WITH {:?} & {:?}", type1, type2);
            if *type1 == ObjectType::PlayerFeet || *type2 == ObjectType::PlayerFeet {
                // The player feets touched the ground.
                // That means we are grounded.
                ground = true;
                //info!("GROUNDED");
            }
        }

        if let Some(ground_comp) = (&players, &mut groundeds).join().next().map(|t| t.1) {
            if ground && !ground_comp.ground {
                // Just grounded
                ground_comp.since = time.absolute_time_seconds();
            }
            ground_comp.ground = ground;
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

pub fn avg_float_to_string(value: f32, decimals: u32) -> String {
    let mult = 10.0_f32.powf(decimals as f32);
    ((value * mult).ceil() / mult).to_string()
}

pub fn add_removal_to_entity(entity: Entity, id: RemovalId, world: &World) {
	world
        .write_storage::<Removal<RemovalId>>()
        .insert(entity, Removal::new(id))
        .expect(&format!("Failed to insert removalid to entity {:?}.", entity));
}

/*pub struct FutureProcessor<'a, F> where F: SystemData<'a> {
    pub queue: Arc<Mutex<VecDeque<Box<Fn(F)>>>>,
    pub phantom: &'a PhantomData<()>,
}*/

type FutureQueue = Arc<Mutex<VecDeque<Box<Fn(&mut World) + Send>>>>;

/// Resource holding the queue of results functions.
/// Filled when a Future finishes its work (or fails).
/// Emptied by the amethyst main loop.
#[derive(Default)]
pub struct FutureProcessor {
    pub queue: FutureQueue,
}

impl FutureProcessor {
    pub fn queue_ref(&self) -> FutureQueue {
        self.queue.clone()
    }
}

pub fn do_login(future_runtime: &mut Runtime, queue: &FutureProcessor, username: String, password: String) {
    let https = HttpsConnector::new(4).expect("TLS initialization failed");
    let client = Client::builder().build::<_, hyper::Body>(https);
    let request = Request::post("https://hoppinworld.net:27015/login")
        .header("Content-Type", "application/json")
        .body(Body::from(format!("{{\"email\":\"{}\", \"password\":\"{}\"}}", username, password)))
        .unwrap();

    let queue_ref = queue.queue_ref();
    let future = client
        // Fetch the url...
        .request(request)
        // And then, if we get a response back...
        .and_then(move |result| {
            println!("Response: {}", result.status());
            println!("Headers: {:#?}", result.headers());

            // The body is a stream, and for_each returns a new Future
            // when the stream is finished, and calls the closure on
            // each chunk of the body...
            result.into_body().for_each(move |chunk| {
                /*io::stdout().write_all(&chunk)
                    .map_err(|e| panic!("example expects stdout is open, error={}", e))*/
                match serde_json::from_slice::<Auth>(&chunk) {
                    Ok(a) => queue_ref.lock().unwrap().push_back(Box::new(move |world| {
                        world.add_resource(a.clone());
                    })),
                    Err(e) => eprintln!("Failed to parse received data to Auth: {}", e),
                }
                Ok(())
            })
            //serde_json::from_slice::<Auth>(result.into_body())
        })
        // If all good, just tell the user...
        .map(move |_| {
            println!("\n\nDone.");
            /*queue_ref.lock().unwrap().push_back(Box::new(move |world| {
                world.add_resource(auth);
            }));*/
        })
        // If there was an error, let the user know...
        .map_err(|err| {
            eprintln!("Error {}", err);
        });
    //tokio::run(future);
    /*while let Ok(Async::NotReady) = future.poll() {
        println!("POLLING");
    }*/
    //println!("Done");
    future_runtime.spawn(future);

    //runtime.shutdown_on_idle().wait().unwrap();
}

// TODO remove dup from backend
#[derive(Serialize, Deserialize)]
pub struct ScoreInsertRequest {
    pub mapid: i32,
    pub segment_times: Vec<f32>,
    pub strafes: i32,
    pub jumps: i32,
    /// Seconds
    pub total_time: f32,
    pub max_speed: f32,
    pub average_speed: f32,
}

pub fn submit_score(future_runtime: &mut Runtime, queue: &FutureProcessor, auth_token: String, score_insert_request: ScoreInsertRequest) {
    let https = HttpsConnector::new(4).expect("TLS initialization failed");
    let client = Client::builder().build::<_, hyper::Body>(https);
    let request = Request::post("https://hoppinworld.net:27015/submitscore")
        .header("Content-Type", "application/json")
        .header("X-Authorization", format!("Bearer {}", auth_token))
        .body(Body::from(json!(score_insert_request).to_string()))
        .unwrap();

    let queue_ref = queue.queue_ref();
    let future = client
        // Fetch the url...
        .request(request)
        // And then, if we get a response back...
        .and_then(move |result| {
            println!("Response: {}", result.status());
            println!("Headers: {:#?}", result.headers());

            result.into_body().for_each(move |chunk| {
                /*match serde_json::from_slice::<Auth>(&chunk) {
                    Ok(a) => queue_ref.lock().unwrap().push_back(Box::new(move |world| {
                        world.add_resource(a.clone());
                    })),
                    Err(e) => eprintln!("Failed to parse received data to Auth: {}", e),
                }*/
                info!("{}", String::from_utf8(chunk.to_vec()).unwrap_or("~~Failure to convert server answer to string~~".to_string()));
                Ok(())
            })
        })
        // If all good, just tell the user...
        .map(move |_| {
            println!("\n\nDone submitting score.");
        })
        .map_err(|err| {
            eprintln!("Error {}", err);
        });
    future_runtime.spawn(future);
}


pub fn verts_from_mesh_data(mesh_data: &MeshData, scale: &Vector3<f32>) -> Vec<Point3<f32>> {
	if let MeshData::Creator(combo) = mesh_data {
        combo
            .vertices()
            .iter()
            .map(|sep| {
                Point3::new(
                    (sep.0)[0] * scale.x,
                    (sep.0)[1] * scale.y,
                    (sep.0)[2] * scale.z,
                )
            }).collect::<Vec<_>>()
    } else {
    	error!("MeshData was not of combo type! Not extracting vertices.");
        vec![]
    }
}


fn init_discord_rich_presence() -> Result<DiscordRichPresence,()> {
    DiscordRichPresence::new(498979571933380609, "Main Menu".to_string(), Some("large_image".to_string()), Some("Hoppin World".to_string()), None, None)
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

    /*amethyst::start_logger(amethyst::LoggerConfig {
		stdout: amethyst::StdoutLog::Colored,
		level_filter: amethyst::LogLevelFilter::Error,
		log_file: None,
		allow_env_override: false,
	});*/

    let resources_directory = get_working_dir();

    let asset_loader = AssetLoader::new(&format!("{}/assets", resources_directory), "base");

    let display_config_path = asset_loader.resolve_path("config/display.ron").unwrap();
    let display_config = DisplayConfig::load(&display_config_path);

    let key_bindings_path = asset_loader.resolve_path("config/input.ron").unwrap();



    // Idea: Show states on StateMachine stack
    // Idea: Time controls (time scale, change manually, etc) core::Time
    // Idea: Clicking on an entity reference inside a component leads to the entity's components
    // Idea: StateEvent<T> history with timestamps
    // Idea: Follow EventChannel. On start, register reader id, then do the same as for StateEvent<T>

    // Issue: If the resource is not present, the game will crash on launch. Solution: Option<Read<T>>
    // issue thread '<unnamed>' panicked at 'Failed to send message: Os { code: 90, kind: Other, message: "Message too long" }', libcore/result.rs:1009:5
    // Issue: Laggy as hell. 34 entites, 150 components
    // Issue: thread '<unnamed>' panicked at 'Failed to send message: Os { code: 111, kind: ConnectionRefused, message: "Connection refused" }
    //   a.k.a can't run without the editor open, which is not really convenient ^^

    /*let editor_bundle = SyncEditorBundle::new()
    .sync_component::<Transform>("Transform")
    .sync_component::<BodyPose3<f32>>("BodyPose")
    .sync_component::<UiTransform>("UiTransform")
    //.sync_component::<MeshData>("MeshData") // Bug: Failed to serialize
    .sync_component::<UiText>("UiText")
    .sync_component::<Removal<RemovalId>>("Removal")
    .sync_component::<Mass3<f32>>("Mass")
    .sync_component::<Velocity3<f32>>("Velocity")
    .sync_component::<NextFrame<Velocity3<f32>>>("NextVelocity")
    .sync_component::<NextFrame<BodyPose3<f32>>>("NextBodyPose")
    .sync_component::<ObjectType>("Collider:ObjectType")
    .sync_component::<BhopMovement3D>("BhopMovement3D")
    .sync_component::<Player>("Player")
    .sync_component::<UiButton>("UiButton")
    .sync_component::<FlyControlTag>("FlyControlTag")
    .sync_component::<Shape>("Shape")
    .sync_component::<ForceAccumulator<f32, f32>>("ForceAccumulator")
    .sync_component::<RotationControl>("RotationControl")
    .sync_component::<Camera>("Camera")
    //.sync_component::<Parent>("Parent")
    .sync_component::<Light>("Light")
    .sync_component::<Named>("Named")
    //.sync_component::<Primitive3<f32>>("Collider:Primitive")
    .sync_resource::<Gravity>("Gravity")
    .sync_resource::<RelativeTimer>("RelativeTimer")
    .sync_resource::<RuntimeProgress>("RuntimeProgress")
    //.sync_resource::<RuntimeStats>("RuntimeStats") // Not present on game start
    //.sync_resource::<RuntimeMap>("RuntimeMap")
    .sync_resource::<AmbientColor>("AmbientColor")
    //.sync_resource::<WorldParameters<f32,f32>>("WorldParameters") // Not present on game start
    .sync_resource::<MapInfoCache>("MapInfoCache")
    .sync_resource::<HideCursor>("HideCursor")
    ;*/

    let pipe = Pipeline::build().with_stage(
        Stage::with_backbuffer()
            .clear_target([0.1, 0.1, 0.1, 1.0], 1.0)
            .with_pass(
                DrawPbmSeparate::new()
                    .with_transparency(
                    ColorMask::all(),
                    ALPHA,
                    Some(DepthMode::LessEqualWrite),
                )
                /*DrawFlatSeparate::new()
                .with_transparency(
                    ColorMask::all(),
                    ALPHA,
                    Some(DepthMode::LessEqualWrite)
                )*/

            ).with_pass(DrawUi::new()),
    );

    let game_data = GameDataBuilder::default()
        .with(RelativeTimerSystem, "relative_timer", &[])
        .with(
            PrefabLoaderSystem::<ScenePrefab>::default(),
            "map_loader",
            &[],
        )
        .with(
            GltfSceneLoaderSystem::default(),
            "gltf_loader",
            &["map_loader"],
        ).with(
            FPSRotationRhusicsSystem::<String, String>::new(0.3, 0.3),
            "free_rotation",
            &[],
        ).with(MouseFocusUpdateSystem::new(), "mouse_focus", &[])
        .with(CursorHideSystem::new(), "cursor_hide", &[])
        .with(PlayerFeetSync, "feet_sync", &[])
        .with(ColliderGroundedSystem::default(), "ground_checker", &["feet_sync"]) // TODO: Runs one frame late
        // Important to have this after ground checker and before jump.
        .with(JumpSystem::default(), "jump", &["ground_checker"])
        .with(
            GroundFrictionSystem,
            "ground_friction",
            &["ground_checker", "jump"],
        ).with(
            BhopMovementSystem::<String, String>::new(
                Some(String::from("right")),
                Some(String::from("forward")),
            ),
            "bhop_movement",
            &["free_rotation", "jump", "ground_friction", "ground_checker"],
        )
        .with(GravitySystem, "gravity", &[])
        .with_bundle(TransformBundle::new().with_dep(&[]))?
        .with(UiUpdaterSystem, "gameplay_ui_updater", &[])
        .with(ContactSystem::default(), "contacts", &["bhop_movement"])
        .with_bundle(
            InputBundle::<String, String>::new().with_bindings_from_file(&key_bindings_path)?,
        )?.with_bundle(UiBundle::<String, String>::new())?
        .with_barrier()
        .with_bundle(DefaultPhysicsBundle3::<ObjectType>::new().with_spatial())?
        .with_bundle(RenderBundle::new(pipe, Some(display_config))
            //.with_visibility_sorting(&[])
        )?
        .with_bundle(FPSCounterBundle)?
        //.with_bundle(editor_bundle)?
        ;

    let mut game_builder = CoreApplication::<_, AllEvents, AllEventsReader>::build(resources_directory, InitState::default())?
        .with_resource(asset_loader)
        .with_resource(AssetLoaderInternal::<FontAsset>::new())
        .with_resource(AssetLoaderInternal::<Prefab<GltfPrefab>>::new());
    if let Ok(discord) = init_discord_rich_presence() {
        game_builder = game_builder.with_resource(discord);
    }
    let mut game = game_builder.build(game_data)?;
    game.run();
    Ok(())
}
