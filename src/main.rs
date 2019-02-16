#[macro_use]
extern crate amethyst;
//extern crate amethyst_core;
extern crate amethyst_extra;
extern crate amethyst_gltf;
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
extern crate amethyst_editor_sync;
extern crate crossbeam_channel;
extern crate hoppinworlddata;
extern crate hoppinworldruntime;
extern crate hyper;
extern crate hyper_tls;
extern crate num_traits;
extern crate ron;
extern crate tokio;
extern crate tokio_executor;
extern crate uuid;

/*#[macro_use]
extern crate derive_builder;*/

use amethyst::assets::Prefab;
use amethyst::assets::PrefabLoaderSystem;
use amethyst::controls::*;
use amethyst::controls::{CursorHideSystem, MouseFocusUpdateSystem};
use amethyst::core::nalgebra::Point3;
use amethyst::core::transform::TransformBundle;
use amethyst::core::{Named, Time, Transform};
use amethyst::ecs::{
    Entities, Entity, Join, Read, ReadStorage, Resources, System, SystemData, Write, WriteStorage,
};
use amethyst::input::InputBundle;
use amethyst::prelude::*;
use amethyst::renderer::{
    AmbientColor, Camera, ColorMask, DepthMode, DisplayConfig, DrawPbmSeparate, Light, MeshData,
    Pipeline, RenderBundle, Stage, ALPHA,
};
use amethyst::shrev::{EventChannel, ReaderId};
use amethyst::ui::*;
use amethyst::utils::application_root_dir;
use amethyst::utils::removal::Removal;
use amethyst_editor_sync::*;
use amethyst_extra::nphysics_ecs::ncollide::events::ProximityEvent;
use amethyst_extra::nphysics_ecs::*;
use amethyst_gltf::*;
use crossbeam_channel::Sender;
use hoppinworldruntime::*;
use amethyst_extra::dirty::Dirty;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use amethyst::core::nalgebra::Vector3;
use amethyst::utils::fps_counter::FPSCounterBundle;
use amethyst_extra::*;
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

pub fn do_login(
    future_runtime: &mut Runtime,
    queue: Sender<Callback>,
    username: String,
    password: String,
) {
    let https = HttpsConnector::new(2).expect("TLS initialization failed");
    let client = Client::builder().build::<_, hyper::Body>(https);
    let request = Request::post("https://hoppinworld.net:27015/login")
        .header("Content-Type", "application/json")
        .body(Body::from(format!(
            "{{\"email\":\"{}\", \"password\":\"{}\"}}",
            username, password
        )))
        .unwrap();

    let future = client
        .request(request)
        .and_then(move |result| {
            println!("Response: {}", result.status());
            println!("Headers: {:#?}", result.headers());
            result.into_body().for_each(move |chunk| {
                match serde_json::from_slice::<Auth>(&chunk) {
                    Ok(a) => queue
                        .send(Box::new(move |world| {
                            world.write_resource::<Dirty<Auth>>().write().token = a.token.clone();
                            world.write_resource::<Dirty<Auth>>().write().set_validated(true);
                        }))
                        .expect("Failed to push auth callback to future queue"),
                    Err(e) => error!("Failed to parse received data to Auth: {}", e),
                }
                Ok(())
            })
        })
        .map(move |_| {
            info!("\n\nLogin successful.");
        })
        // TODO: Show error
        .map_err(|err| {
            error!("Failed to login. Error: {}", err);
        });
    future_runtime.spawn(future);
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

pub fn submit_score(
    future_runtime: &mut Runtime,
    auth_token: String,
    score_insert_request: ScoreInsertRequest,
) {
    let https = HttpsConnector::new(4).expect("TLS initialization failed");
    let client = Client::builder().build::<_, hyper::Body>(https);
    let request = Request::post("https://hoppinworld.net:27015/submitscore")
        .header("Content-Type", "application/json")
        .header("X-Authorization", format!("Bearer {}", auth_token))
        .body(Body::from(json!(score_insert_request).to_string()))
        .unwrap();

    let future = client
        .request(request)
        .and_then(move |result| {
            println!("Response: {}", result.status());
            println!("Headers: {:#?}", result.headers());

            result.into_body().for_each(move |chunk| {
                info!(
                    "{}",
                    String::from_utf8(chunk.to_vec())
                        .unwrap_or("Error converting server answer to string after score submission".to_string())
                );
                Ok(())
            })
        })
        .map(move |_| {
            info!("\n\nScore submitted with success.0");
        })
        .map_err(|err| {
            error!("Error {}", err);
        });
    future_runtime.spawn(future);
}

pub fn validate_auth_token(
    future_runtime: &mut Runtime,
    auth_token: String,
    queue: Sender<Callback>,
) {
    let https = HttpsConnector::new(4).expect("TLS initialization failed");
    let client = Client::builder().build::<_, hyper::Body>(https);
    let request = Request::post("https://hoppinworld.net:27015/validatetoken")
        .header("Content-Type", "application/json")
        .header("X-Authorization", format!("Bearer {}", auth_token))
        .body(Body::from(json!(auth_token).to_string()))
        .unwrap();

    let future = client
        .request(request)
        .and_then(move |result| {
            println!("Response: {}", result.status());
            println!("Headers: {:#?}", result.headers());

            result.into_body().for_each(move |chunk| {
                let valid = match serde_json::from_slice::<bool>(&chunk) {
                    Ok(a) => true,
                    Err(e) => {
                        error!("Failed to parse received data to validation bool: {}", e);
                        false
                    }
                };
                queue
                    .send(Box::new(move |world| {
                        world.write_resource::<Dirty<Auth>>().write().set_validated(valid);
                    }))
                    .expect("Failed to push auth validation callback to future queue");
                Ok(())
            })
        })
        .map(move |_| {
            info!("\n\nAuth token validation request submitted with success to server.");
        })
        .map_err(|err| {
            error!("Error {}", err);
        });
    future_runtime.spawn(future);
}


fn init_discord_rich_presence() -> Result<DiscordRichPresence, ()> {
    DiscordRichPresence::new(
        498979571933380609,
        "Main Menu".to_string(),
        Some("large_image".to_string()),
        Some("Hoppin World".to_string()),
        None,
        None,
    )
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

    if cfg!(debug_assertions) {
        amethyst::start_logger(Default::default());
    } else {
        amethyst::start_logger(amethyst::LoggerConfig {
            stdout: amethyst::StdoutLog::Colored,
            level_filter: amethyst::LogLevelFilter::Error,
            log_file: None,// TODO some
            allow_env_override: false,
        });
    }

    let mut resources_directory = application_root_dir().expect("Failed to get app_root_dir.");
    resources_directory.push("assets");

    let asset_loader = AssetLoader::new(&resources_directory.to_str().unwrap(), "base");

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

    /*let components = type_set![Transform, UiTransform, UiText, Removal<RemovalId>, ObjectType, BhopMovement3D, UiButton, FlyControlTag,RotationControl, Camera,Light, Named];

    let editor_bundle = SyncEditorBundle::new()
    .sync_components(&components)
    //.sync_component::<Primitive3<f32>>("Collider:Primitive")
    .sync_resource::<Gravity>("Gravity")
    //.sync_resource::<RelativeTimer>("RelativeTimer")
    //.sync_resource::<RuntimeProgress>("RuntimeProgress")
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
                DrawPbmSeparate::new().with_transparency(
                    ColorMask::all(),
                    ALPHA,
                    Some(DepthMode::LessEqualWrite),
                ), /*DrawFlatSeparate::new()
                   .with_transparency(
                       ColorMask::all(),
                       ALPHA,
                       Some(DepthMode::LessEqualWrite)
                   )*/
            )
            .with_pass(DrawUi::new()),
    );

    let noclip = NoClip::new(String::from("noclip"));

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
            FPSRotationRhusicsSystem::<String, String>::new(0.005, 0.005),
            "free_rotation",
            &[],
        ).with(MouseFocusUpdateSystem::new(), "mouse_focus", &[])
        .with(CursorHideSystem::new(), "cursor_hide", &[])
        .with(PlayerFeetSync, "feet_sync", &[])
        // runs one frame late?
        .with(GroundCheckerSystem::new(Vec::<ObjectType>::new()), "ground_checker", &["feet_sync"])
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
        .with(UiUpdaterSystem, "gameplay_ui_updater", &[])
        .with(ContactSystem::default(), "contacts", &["bhop_movement"])
        .with_bundle(TransformBundle::new().with_dep(&["free_rotation", "feet_sync", "contacts"]))?
        //.with(NoClipToggleSystem::<String>::default(), "noclip_toggle", &[])
        //.with(FreeRotationSystem::<String, String>::new(0.03, 0.03), "noclip_rotation", &[])
        //.with(FlyMovementSystem::<String, String>::new(6.0, Some("right".to_string()), Some("up".to_string()), Some("forward".to_string())), "fly_movement", &[])
        .with_bundle(
            InputBundle::<String, String>::new().with_bindings_from_file(&key_bindings_path)?,
        )?.with_bundle(UiBundle::<String, String>::new())?
        .with(AutoSaveSystem::<Auth>::new(resources_directory.to_str().unwrap().to_owned() + "/../auth_token.ron"), "auth_token_save", &[])        .with_barrier()
        .with_bundle(PhysicsBundle::new())?
        //.with(ForceUprightSystem::default(), "force_upright", &["sync_bodies_from_physics_system"])
        .with_bundle(RenderBundle::new(pipe, Some(display_config))
            //.with_visibility_sorting(&[])
        )?
        .with_bundle(FPSCounterBundle)?
        //.with_bundle(editor_bundle)?
        ;

    let mut game_builder = CoreApplication::<_, AllEvents, AllEventsReader>::build(
        resources_directory,
        InitState::default(),
    )?
    .with_resource(asset_loader)
    .with_resource(AssetLoaderInternal::<FontAsset>::new())
    .with_resource(AssetLoaderInternal::<Prefab<GltfPrefab>>::new())
    .with_resource(noclip)
    .with_resource(TimeStep::Fixed(1./120.));
    if let Ok(discord) = init_discord_rich_presence() {
        game_builder = game_builder.with_resource(discord);
    }
    let mut game = game_builder.build(game_data)?;
    game.run();
    Ok(())
}
