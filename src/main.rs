extern crate amethyst;
extern crate amethyst_extra;
extern crate amethyst_gltf;
extern crate amethyst_rhusics;
#[macro_use]
extern crate serde;
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

use amethyst::assets::{
    AssetPrefab, PrefabData, PrefabLoader, PrefabLoaderSystem, ProgressCounter,
};
use amethyst::controls::{
    CursorHideSystem, FlyControlTag, HideCursor, MouseFocusUpdateSystem, WindowFocus,
};
use amethyst::core::transform::{Parent, Transform, TransformBundle};
use amethyst::core::{Named, Time};
use amethyst::ecs::error::Error as ECSError;
use amethyst::ecs::prelude::ParallelIterator;
use amethyst::ecs::{
    Component, DenseVecStorage, Entities, Entity, Join, ParJoin, Read, ReadStorage, Resources,
    System, SystemData, Write, WriteStorage,
};
use amethyst::input::{is_key_down, InputBundle, InputHandler};
use amethyst::prelude::*;
use amethyst::renderer::*;
use amethyst::shrev::{EventChannel, ReaderId};
use amethyst::ui::*;
use amethyst::utils::scene::BasicScenePrefab;
use amethyst_gltf::*;
use std::hash::Hash;

use amethyst::core::cgmath::{
    Basis3, Deg, EuclideanSpace, InnerSpace, Matrix3, One, Point3, Quaternion, Rotation3,
    SquareMatrix, Vector2, Vector3,
};
use amethyst_extra::*;
use amethyst_rhusics::collision::dbvt::query_ray;
use amethyst_rhusics::collision::primitive::{ConvexPolyhedron, Cylinder, Primitive3};
use amethyst_rhusics::collision::{Aabb3, Ray3};
use amethyst_rhusics::rhusics_core::physics3d::{Mass3, Velocity3};
use amethyst_rhusics::rhusics_core::{
    Collider, CollisionMode, CollisionShape, CollisionStrategy, ContactEvent, ForceAccumulator,
    Material, NextFrame, PhysicalEntity, Pose,
};
use amethyst_rhusics::rhusics_ecs::physics3d::{BodyPose3, DynamicBoundingVolumeTree3};
use amethyst_rhusics::rhusics_ecs::{PhysicalEntityParts, WithPhysics};
use amethyst_rhusics::{time_sync, DefaultPhysicsBundle3};
use partial_function::*;
use std::marker::PhantomData;
use uuid::Uuid;
use winit::DeviceEvent;

type ScenePrefab = BasicScenePrefab<Vec<PosNormTex>>;
type Shape = CollisionShape<Primitive3<f32>, BodyPose3<f32>, Aabb3<f32>, ObjectType>;
type DefaultPhysicalEntityParts<'a, T> = PhysicalEntityParts<
    'a,
    Primitive3<f32>,
    T,
    Quaternion<f32>,
    Vector3<f32>,
    Vector3<f32>,
    Matrix3<f32>,
    Aabb3<f32>,
    BodyPose3<f32>,
>;
type MyPhysicalEntityParts<'a> = DefaultPhysicalEntityParts<'a, ObjectType>;
type CustomTrans<'a, 'b> = Trans<GameData<'a, 'b>, CustomStateEvent>;

const DISPLAY_SPEED_MULTIPLIER: f32 = 50.0;

#[derive(PartialEq, Eq, Debug, Clone, Serialize, Deserialize, Copy)]
pub enum CustomStateEvent {
    // Actually a redirect to MapSelectState in this case.
    GotoMainMenu,
    MapFinished,
}

#[derive(PartialEq, Eq, Debug, Clone, Serialize, Deserialize)]
pub enum RemovalId {
    Scene,
    GameplayUi,
    MenuUi,
    PauseUi,
    ResultUi,
    MapSelectUi,
}

#[repr(u8)]
#[derive(Debug, Clone, PartialOrd, PartialEq, Component)]
pub enum ObjectType {
    Scene,
    StartZone,
    EndZone,
    KillZone,
    Player,
    SegmentZone(u8),
}

impl Default for ObjectType {
    fn default() -> Self {
        ObjectType::Scene
    }
}

impl Collider for ObjectType {
    fn should_generate_contacts(&self, other: &ObjectType) -> bool {
        let _ret = (*self == ObjectType::Player && *other == ObjectType::Scene)
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


#[derive(Deserialize, Serialize)]
pub struct HoppinMapPrefabData {
    name: String,
    map: AssetPrefab<GltfSceneAsset, GltfSceneFormat>,
}

impl<'a> PrefabData<'a> for HoppinMapPrefabData {
    type SystemData =
        (<AssetPrefab<GltfSceneAsset, GltfSceneFormat> as PrefabData<'a>>::SystemData,);
    type Result = ();

    fn load_prefab(
        &self,
        entity: Entity,
        system_data: &mut Self::SystemData,
        entities: &[Entity],
    ) -> Result<(), ECSError> {
        let (ref mut gltfs,) = system_data;
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


/// Calculates in relative time using the internal engine clock.
#[derive(Default)]
pub struct RelativeTimer {
    pub start: f64,
    pub current: f64,
    pub running: bool,
}

impl RelativeTimer {
    pub fn get_text(&self) -> String {
        ((self.duration() * 1000.0).ceil() / 1000.0).to_string()
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

pub struct RuntimeProgress {
    pub current_segment: u8,
    pub segment_count: u8,
    pub segment_times: Vec<f32>,
}

impl Default for RuntimeProgress {
    fn default() -> Self {
        RuntimeProgress {
            current_segment: 1u8,
            segment_count: 0u8,
            segment_times: vec![],
        }
    }
}

impl RuntimeProgress {
    pub fn new(segment_count: u8) -> Self {
        RuntimeProgress {
            current_segment: 1u8,
            segment_count,
            // +1 to take into account last segment to end zone
            segment_times: vec![0.0; (segment_count + 1) as usize],
        }
    }
}

pub struct RelativeTimerSystem;

impl<'a> System<'a> for RelativeTimerSystem {
    type SystemData = (Write<'a, RelativeTimer>, Read<'a, Time>);
    fn run(&mut self, (mut timer, time): Self::SystemData) {
        timer.update(time.absolute_time_seconds());
    }
}

#[derive(Default, Component, Serialize, Deserialize)]
pub struct Player;

/// Very game dependent.
/// Don't try to make that generic.
#[derive(Default)]
pub struct ContactSystem {
    contact_reader: Option<ReaderId<ContactEvent<Entity, Point3<f32>>>>,
}

impl<'a> System<'a> for ContactSystem {
    type SystemData = (
        Entities<'a>,
        Read<'a, EventChannel<ContactEvent<Entity, Point3<f32>>>>,
        Write<'a, RelativeTimer>,
        Read<'a, Time>,
        ReadStorage<'a, ObjectType>,
        ReadStorage<'a, Player>,
        ReadStorage<'a, BhopMovement3D>,
        WriteStorage<'a, NextFrame<Velocity3<f32>>>,
        Write<'a, EventChannel<CustomStateEvent>>,
        Write<'a, RuntimeProgress>,
    );

    fn run(
        &mut self,
        (
            entities,
            contacts,
            mut timer,
            time,
            object_types,
            players,
            bhop_movements,
            mut velocities,
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
                    timer.start(time.absolute_time_seconds());
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
                    runtime_progress.segment_times[id] = timer.duration() as f32;
                    state_eventchannel.single_write(CustomStateEvent::MapFinished);
                }
                ObjectType::KillZone => {
                    info!("you are ded!");
                }
                ObjectType::SegmentZone(id) => {
                    if *id > runtime_progress.current_segment {
                        runtime_progress.segment_times[(*id - 1) as usize] =
                            timer.duration() as f32;
                        runtime_progress.current_segment = *id;
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

#[derive(Default, Serialize, Deserialize, Clone, Debug)]
pub struct Stats {}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum MapCategory {
    Speed,
    Technical,
    Gimmick,
    Uphill,
    Downhill,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum MapDifficulty {
    Easy,
    Normal,
    Hard,
    Insane,
    Extreme,
    Master,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MapInfo {
    pub id: Uuid,
    pub name: String,
    pub mapper: String,
    pub categories: Vec<MapCategory>,
    pub difficulty: MapDifficulty,
    pub tags: Vec<String>,
}

#[derive(Default, Debug, Clone, Serialize, Deserialize, new)]
pub struct MapInfoCache {
    /// File name -> MapInfo
    pub maps: Vec<(String, MapInfo)>,
}

#[derive(Debug, Clone, new)]
pub struct CurrentMap {
    pub map: (String, MapInfo),
}

pub fn get_all_maps(base_path: &str) -> MapInfoCache {
    let maps_path = format!(
        "{}{}maps{}",
        base_path,
        std::path::MAIN_SEPARATOR,
        std::path::MAIN_SEPARATOR
    );

    let map_info_vec = std::fs::read_dir(&maps_path)
        .expect(&*format!("Failed to read maps directory {}.", &maps_path))
        .filter(|e| e.as_ref().unwrap().file_type().unwrap().is_file())
        .map(|e| e.unwrap().path())
        .filter(|p| {
            p.extension()
                .unwrap_or_else(|| std::ffi::OsStr::new(""))
                .to_str()
                .unwrap()
                == "hop"
        }).map(|e| {
            let info_file_data = std::fs::read_to_string(e.to_str().unwrap()).unwrap();
            let info =
                ron::de::from_str(&info_file_data).expect("Failed to deserialize info map file.");

            Some((e.file_stem().unwrap().to_str().unwrap().to_string(), info))
        }).flatten()
        .collect::<Vec<_>>();
    MapInfoCache::new(map_info_vec)
}

pub fn gltf_path_from_map(base_path: &str, map_name: &str) -> String {
    format!(
        "{}{}maps{}{}.glb",
        base_path,
        std::path::MAIN_SEPARATOR,
        std::path::MAIN_SEPARATOR,
        map_name
    )
}

/// Very game dependent.
pub struct UiUpdaterSystem;

impl<'a> System<'a> for UiUpdaterSystem {
    type SystemData = (
        Read<'a, RelativeTimer>,
        Read<'a, Stats>,
        ReadStorage<'a, Velocity3<f32>>,
        ReadStorage<'a, Jump>,
        ReadStorage<'a, UiTransform>,
        WriteStorage<'a, UiText>,
        ReadStorage<'a, Player>,
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

pub fn avg_float_to_string(value: f32, decimals: u32) -> String {
    let mult = 10.0_f32.powf(decimals as f32);
    ((value * mult).ceil() / mult).to_string()
}

#[derive(Default)]
struct InitState;

impl<'a, 'b> State<GameData<'a, 'b>, CustomStateEvent> for InitState {
    fn on_start(&mut self, data: StateData<GameData>) {
        data.world.register::<ObjectType>();
        data.world.register::<Removal<RemovalId>>();
        data.world.add_resource(get_all_maps(&get_working_dir()));
    }

    fn update(&mut self, data: StateData<GameData>) -> CustomTrans<'a, 'b> {
        data.data.update(&data.world);
        Trans::Switch(Box::new(MainMenuState))
    }
}

#[derive(Default)]
struct MainMenuState;

impl<'a, 'b> State<GameData<'a, 'b>, CustomStateEvent> for MainMenuState {
    fn on_start(&mut self, data: StateData<GameData>) {
        let hide_cursor = HideCursor { hide: false };
        data.world.add_resource(hide_cursor);

        data.world.register::<ObjectType>();
        data.world.register::<Removal<RemovalId>>();

        let ui_root = data
            .world
            .exec(|mut creator: UiCreator| creator.create("assets/base/prefabs/menu_ui.ron", ()));
        data.world
            .write_storage::<Removal<RemovalId>>()
            .insert(ui_root, Removal::new(RemovalId::MenuUi))
            .expect("Failed to insert removalid to ui_root for main menu state.");
    }

    fn update(&mut self, data: StateData<GameData>) -> CustomTrans<'a, 'b> {
        data.data.update(&data.world);
        Trans::None
    }

    fn handle_event(
        &mut self,
        data: StateData<GameData>,
        event: StateEvent<CustomStateEvent>,
    ) -> CustomTrans<'a, 'b> {
        match event {
            StateEvent::Ui(UiEvent {
                event_type: UiEventType::Click,
                target: entity,
            }) => {
                if let Some(ui_transform) = data.world.read_storage::<UiTransform>().get(entity) {
                    match &*ui_transform.id {
                        "play_button" => Trans::Switch(Box::new(MapSelectState::default())),
                        "quit_button" => Trans::Quit,
                        _ => Trans::None,
                    }
                } else {
                    Trans::None
                }
            }
            _ => Trans::None,
        }
    }

    fn on_stop(&mut self, data: StateData<GameData>) {
        exec_removal(
            &data.world.entities(),
            &data.world.read_storage(),
            RemovalId::MenuUi,
        );
    }
}

#[derive(Default)]
struct MapSelectState;

impl<'a, 'b> State<GameData<'a, 'b>, CustomStateEvent> for MapSelectState {
    fn on_start(&mut self, data: StateData<GameData>) {
        let ui_root = data.world.exec(|mut creator: UiCreator| {
            creator.create("assets/base/prefabs/map_select_ui.ron", ())
        });
        data.world
            .write_storage::<Removal<RemovalId>>()
            .insert(ui_root, Removal::new(RemovalId::MapSelectUi))
            .expect("Failed to insert removalid to ui_root for map select state.");

        let font = data
            .world
            .read_resource::<AssetLoader>()
            .load(
                "font/arial.ttf",
                FontFormat::Ttf,
                (),
                &mut data.world.write_resource(),
                &mut data.world.write_resource(),
                &data.world.read_resource(),
            ).expect("Failed to load font");
        let maps = data.world.read_resource::<MapInfoCache>().maps.clone();
        for (accum, (internal, info)) in maps.into_iter().enumerate() {
            info!("adding map!");
            let entity =
                UiButtonBuilder::new(format!("map_select_{}", internal), info.name.clone())
                    .with_font(font.clone())
                    .with_text_color([0.2, 0.2, 0.2, 1.0])
                    .with_font_size(30.0)
                    .with_size(512.0, 200.0)
                    .with_layer(8.0)
                    .with_position(0.0, -300.0 - 100.0 * accum as f32)
                    .with_anchor(Anchor::TopMiddle)
                    .build_from_world(data.world);
            data.world
                .write_storage::<Removal<RemovalId>>()
                .insert(entity, Removal::new(RemovalId::MapSelectUi))
                .unwrap();
        }
    }

    fn update(&mut self, data: StateData<GameData>) -> CustomTrans<'a, 'b> {
        data.data.update(&data.world);
        Trans::None
    }

    fn handle_event(
        &mut self,
        data: StateData<GameData>,
        event: StateEvent<CustomStateEvent>,
    ) -> CustomTrans<'a, 'b> {
        let mut change_map = None;
        match event {
            StateEvent::Ui(UiEvent {
                event_type: UiEventType::Click,
                target: entity,
            }) => {
                if let Some(ui_transform) = data.world.read_storage::<UiTransform>().get(entity) {
                    match &*ui_transform.id {
                        "back_button" => {
                            return Trans::Switch(Box::new(MainMenuState::default()));
                        }
                        id => {
                            if id.starts_with("map_select_") {
                                let map_name = &id[11..];
                                change_map = Some(
                                    data.world
                                        .read_resource::<MapInfoCache>()
                                        .maps
                                        .iter()
                                        .find(|t| t.0 == map_name)
                                        .unwrap()
                                        .clone(),
                                );
                            }
                        }
                    }
                }
            }
            StateEvent::Window(ev) => {
                if is_key_down(&ev, VirtualKeyCode::Escape) {
                    return Trans::Switch(Box::new(MainMenuState::default()));
                }
            }
            _ => {}
        }

        if let Some(row) = change_map {
            data.world.add_resource(CurrentMap::new(row));
            return Trans::Switch(Box::new(MapLoadState::default()));
        }
        Trans::None
    }

    fn on_stop(&mut self, data: StateData<GameData>) {
        exec_removal(
            &data.world.entities(),
            &data.world.read_storage(),
            RemovalId::MapSelectUi,
        );
    }
}

#[derive(Default)]
struct GameplayState;

impl<'a, 'b> State<GameData<'a, 'b>, CustomStateEvent> for GameplayState {
    fn on_start(&mut self, data: StateData<GameData>) {
        data.world.write_resource::<HideCursor>().hide = true;
        data.world.write_resource::<Time>().set_time_scale(1.0);
    }

    fn update(&mut self, data: StateData<GameData>) -> CustomTrans<'a, 'b> {
        time_sync(&data.world);
        data.data.update(&data.world);
        Trans::None
    }

    fn handle_event(
        &mut self,
        _data: StateData<GameData>,
        event: StateEvent<CustomStateEvent>,
    ) -> CustomTrans<'a, 'b> {
        // TODO: Map finished
        match event {
            StateEvent::Window(ev) => {
                if is_key_down(&ev, VirtualKeyCode::Escape) {
                    Trans::Push(Box::new(PauseMenuState::default()))
                } else {
                    Trans::None
                }
            }
            StateEvent::Custom(CustomStateEvent::GotoMainMenu) => {
                Trans::Switch(Box::new(MapSelectState::default()))
            }
            StateEvent::Custom(CustomStateEvent::MapFinished) => {
                Trans::Switch(Box::new(ResultState::default()))
            }
            _ => Trans::None,
        }
    }

    fn on_pause(&mut self, data: StateData<GameData>) {
        data.world.write_resource::<HideCursor>().hide = false;
        data.world.write_resource::<Time>().set_time_scale(0.0);
    }

    fn on_resume(&mut self, data: StateData<GameData>) {
        data.world.write_resource::<HideCursor>().hide = true;
        data.world.write_resource::<Time>().set_time_scale(1.0);
    }

    fn on_stop(&mut self, data: StateData<GameData>) {
        data.world.write_resource::<HideCursor>().hide = false;
        // Not sure if I should put 0. Might cause errors later when implementing replays and stuff.
        data.world.write_resource::<Time>().set_time_scale(1.0);
        exec_removal(
            &data.world.entities(),
            &data.world.read_storage(),
            RemovalId::Scene,
        );
        exec_removal(
            &data.world.entities(),
            &data.world.read_storage(),
            RemovalId::GameplayUi,
        );
    }
}

#[derive(Default)]
struct PauseMenuState;

impl<'a, 'b> State<GameData<'a, 'b>, CustomStateEvent> for PauseMenuState {
    fn on_start(&mut self, data: StateData<GameData>) {
        let ui_root = data
            .world
            .exec(|mut creator: UiCreator| creator.create("assets/base/prefabs/pause_ui.ron", ()));
        data.world
            .write_storage::<Removal<RemovalId>>()
            .insert(ui_root, Removal::new(RemovalId::PauseUi))
            .expect("Failed to insert removalid to ui_root for pause state.");
    }

    fn update(&mut self, data: StateData<GameData>) -> CustomTrans<'a, 'b> {
        // Necessary otherwise rhusics will keep the same DeltaTime and will not be paused.
        time_sync(&data.world);
        data.data.update(&data.world);
        Trans::None
    }

    fn handle_event(
        &mut self,
        data: StateData<GameData>,
        event: StateEvent<CustomStateEvent>,
    ) -> CustomTrans<'a, 'b> {
        match event {
            StateEvent::Ui(UiEvent {
                event_type: UiEventType::Click,
                target: entity,
            }) => {
                if let Some(ui_transform) = data.world.read_storage::<UiTransform>().get(entity) {
                    match &*ui_transform.id {
                        "resume_button" => Trans::Pop,
                        "quit_button" => {
                            data.world
                                .write_resource::<EventChannel<CustomStateEvent>>()
                                .single_write(CustomStateEvent::GotoMainMenu);
                            Trans::Pop
                        }
                        _ => Trans::None,
                    }
                } else {
                    Trans::None
                }
            }
            StateEvent::Window(ev) => {
                if is_key_down(&ev, VirtualKeyCode::Escape) {
                    Trans::Pop
                } else {
                    Trans::None
                }
            }
            _ => Trans::None,
        }
    }

    fn on_stop(&mut self, data: StateData<GameData>) {
        exec_removal(
            &data.world.entities(),
            &data.world.read_storage(),
            RemovalId::PauseUi,
        );
    }
}

#[derive(Default)]
struct MapLoadState {
    load_progress: Option<ProgressCounter>,
    init_done: bool,
}

impl<'a, 'b> State<GameData<'a, 'b>, CustomStateEvent> for MapLoadState {
    fn on_start(&mut self, data: StateData<GameData>) {
        data.world.write_resource::<Time>().set_time_scale(0.0);
        self.init_done = false;

        let mut pg = ProgressCounter::new();

        let name = data.world.read_resource::<CurrentMap>().map.0.clone();
        let scene_handle = data.world.exec(|loader: PrefabLoader<GltfPrefab>| {
            loader.load(
                gltf_path_from_map(&get_working_dir(), &name),
                GltfSceneFormat,
                GltfSceneOptions::default(),
                &mut pg,
            )
        });
        self.load_progress = Some(pg);

        let scene_root = data.world.create_entity().with(scene_handle).build();
        data.world
            .write_storage::<Removal<RemovalId>>()
            .insert(scene_root, Removal::new(RemovalId::Scene))
            .expect("Failed to insert removalid to scene for gameplay state.");

        data.world
            .add_resource(Gravity::new(Vector3::new(0.0, -2.0, 0.0)));

        let mut tr = Transform::default();
        tr.translation = [0.0, 5.0, 0.0].into();

        let movement = BhopMovement3D::new(false, 20.0, 20.0, 2.0, 0.5, true);
        let ground_friction = GroundFriction3D::new(2.0, FrictionMode::Percent, 0.15);

        let player_entity = data
            .world
            .create_entity()
            .with(FlyControlTag)
            .with(Grounded::new(0.5))
            .with(ObjectType::Player)
            .with(movement)
            .with(ground_friction)
            .with(Jump::new(true, true, 50.0, true))
            .with(Player)
            .with_dynamic_physical_entity(
                Shape::new_simple_with_type(
                    CollisionStrategy::FullResolution,
                    CollisionMode::Discrete,
                    Cylinder::new(0.5, 0.2).into(),
                    ObjectType::Player,
                ),
                BodyPose3::new(
                    Point3::new(tr.translation.x, tr.translation.y, tr.translation.z),
                    Quaternion::<f32>::one(),
                ),
                Velocity3::default(),
                PhysicalEntity::new(Material::new(1.0, 0.05)),
                Mass3::new(1.0),
            ).with(tr)
            .with(ForceAccumulator::<Vector3<f32>, Vector3<f32>>::new())
            .with(Removal::new(RemovalId::Scene))
            .build();

        let mut tr = Transform::default();
        // TODO add conf ability to this
        tr.translation = [0.0, 0.35, 0.0].into();
        data.world
            .create_entity()
            .with(tr)
            .with(RotationControl::default())
            .with(Camera::standard_3d(1920.0, 1080.0))
            .with(Parent {
                entity: player_entity,
            }).build();

        data.world.add_resource(AmbientColor(Rgba::from([0.1; 3])));
        let mut tr = Transform::default();
        tr.translation = [0.0, 10.0, 0.0].into();
        tr.rotation = Quaternion::one();
        data.world
            .create_entity()
            .with(tr)
            .with(Light::Point(PointLight {
                color: Rgba::white(),
                intensity: 10.0,
                radius: 60.0,
                smoothness: 4.0,
            })).with(Removal::new(RemovalId::Scene))
            .build();

        let ui_root = data.world.exec(|mut creator: UiCreator| {
            creator.create("assets/base/prefabs/gameplay_ui.ron", ())
        });
        data.world
            .write_storage::<Removal<RemovalId>>()
            .insert(ui_root, Removal::new(RemovalId::GameplayUi))
            .expect("Failed to insert removalid to ui_root for gameplay state.");

        data.world.add_resource(RuntimeProgress::default());
        MyPhysicalEntityParts::setup(&mut data.world.res)
    }

    fn update(&mut self, data: StateData<GameData>) -> CustomTrans<'a, 'b> {
        if !self.init_done && self.load_progress.as_ref().unwrap().is_complete() {
            info!("BBBBBBBBBBBBBBBBBBBBBBBBBBBBB");
            let entity_sizes = (
                &*data.world.entities(),
                &data.world.read_storage::<Transform>(),
                &data.world.read_storage::<MeshData>(),
                &data.world.read_storage::<Named>(),
            )
                .par_join()
                .map(|(entity, transform, mesh_data, name)| {
                    info!("map transform: {:?}", transform);
                    let verts = if let MeshData::Creator(combo) = mesh_data {
                        info!("vertices: {:?}", combo.vertices());
                        combo
                            .vertices()
                            .iter()
                            .map(|sep| {
                                Point3::new(
                                    (sep.0)[0] * transform.scale.x,
                                    (sep.0)[1] * transform.scale.y,
                                    (sep.0)[2] * transform.scale.z,
                                )
                            }).collect::<Vec<_>>()
                    } else {
                        vec![]
                    };
                    (entity, transform.clone(), verts, name.name.clone())
                }).collect::<Vec<_>>();

            if !entity_sizes.is_empty() {
                // The loading is done, now we add the colliders.

                let mut max_segment = 0;
                self.init_done = true;
                {
                    let mut collider_storage = data.world.write_storage::<ObjectType>();
                    for (entity, _, _, name) in &entity_sizes {
                        let (obj_type, _coll_strat) = if name == "StartZone" {
                            (ObjectType::StartZone, CollisionStrategy::CollisionOnly)
                        } else if name == "EndZone" {
                            (ObjectType::EndZone, CollisionStrategy::CollisionOnly)
                        } else if name.starts_with("KillZone") {
                            (ObjectType::KillZone, CollisionStrategy::CollisionOnly)
                        } else if name.starts_with("SegmentZone") {
                            let id_str = &name[11..];
                            let id = id_str.to_string().parse::<u8>().unwrap(); // TODO error handling for maps
                            if id > max_segment {
                                max_segment = id;
                            }
                            (
                                ObjectType::SegmentZone(id),
                                CollisionStrategy::CollisionOnly,
                            )
                        } else {
                            (ObjectType::Scene, CollisionStrategy::FullResolution)
                        };
                        collider_storage
                            .insert(*entity, obj_type)
                            .expect("Failed to add ObjectType to map mesh");
                    }
                }
                data.world.add_resource(RuntimeProgress::new(max_segment));
                {
                    let mut physical_parts = MyPhysicalEntityParts::fetch(&data.world.res);
                    for (entity, size, mesh, name) in entity_sizes {
                        let (obj_type, coll_strat) = if name == "StartZone" {
                            (ObjectType::StartZone, CollisionStrategy::CollisionOnly)
                        } else if name == "EndZone" {
                            (ObjectType::EndZone, CollisionStrategy::CollisionOnly)
                        } else if name.starts_with("KillZone") || name.starts_with("SegmentZone") {
                            (ObjectType::KillZone, CollisionStrategy::CollisionOnly)
                        } else {
                            (ObjectType::Scene, CollisionStrategy::FullResolution)
                        };

                        physical_parts
                            .static_entity(
                                entity,
                                Shape::new_simple_with_type(
                                    //CollisionStrategy::FullResolution,
                                    //CollisionStrategy::CollisionOnly,
                                    coll_strat,
                                    CollisionMode::Discrete,
                                    Primitive3::ConvexPolyhedron(<ConvexPolyhedron<f32>>::new(
                                        mesh,
                                    )),
                                    obj_type,
                                ),
                                BodyPose3::new(
                                    Point3::new(
                                        size.translation.x,
                                        size.translation.y,
                                        size.translation.z,
                                    ),
                                    size.rotation,
                                ),
                                PhysicalEntity::default(),
                                Mass3::infinite(),
                            ).expect("Failed to add static collider to map mesh");
                    }
                }
            }
        } else if self.init_done {
            return Trans::Switch(Box::new(GameplayState::default()));
        }

        (&data.world.read_storage::<UiTransform>(),)
            .join()
            .for_each(|tr| info!("ui tr: {:?}", tr));

        data.data.update(&data.world);
        Trans::None
    }
}

#[derive(Default)]
struct ResultState;

impl<'a, 'b> State<GameData<'a, 'b>, CustomStateEvent> for ResultState {
    fn on_start(&mut self, data: StateData<GameData>) {
        let ui_root = data
            .world
            .exec(|mut creator: UiCreator| creator.create("assets/base/prefabs/result_ui.ron", ()));
        data.world
            .write_storage::<Removal<RemovalId>>()
            .insert(ui_root, Removal::new(RemovalId::ResultUi))
            .expect("Failed to insert removalid to ui_root for result state.");
    }

    fn update(&mut self, data: StateData<GameData>) -> CustomTrans<'a, 'b> {
        data.data.update(&data.world);
        Trans::None
    }

    fn handle_event(
        &mut self,
        data: StateData<GameData>,
        event: StateEvent<CustomStateEvent>,
    ) -> CustomTrans<'a, 'b> {
        match event {
            StateEvent::Ui(UiEvent {
                event_type: UiEventType::Click,
                target: entity,
            }) => {
                if let Some(ui_transform) = data.world.read_storage::<UiTransform>().get(entity) {
                    match &*ui_transform.id {
                        "back_button" => Trans::Switch(Box::new(MapSelectState::default())),
                        _ => Trans::None,
                    }
                } else {
                    Trans::None
                }
            }
            StateEvent::Window(ev) => {
                if is_key_down(&ev, VirtualKeyCode::Escape) {
                    Trans::Switch(Box::new(MapSelectState::default()))
                } else {
                    Trans::None
                }
            }
            _ => Trans::None,
        }
    }

    fn on_stop(&mut self, data: StateData<GameData>) {
        exec_removal(
            &data.world.entities(),
            &data.world.read_storage(),
            RemovalId::ResultUi,
        );
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
            .with_pass(DrawPbmSeparate::new().with_transparency(
                ColorMask::all(),
                ALPHA,
                Some(DepthMode::LessEqualWrite),
            )).with_pass(DrawUi::new()),
    );

    let game_data = GameDataBuilder::default()
        .with(RelativeTimerSystem, "relative_timer", &[])
        .with(
            PrefabLoaderSystem::<ScenePrefab>::default(),
            "map_loader",
            &[],
        ).with(
            PrefabLoaderSystem::<HoppinMapPrefabData>::default(),
            "scene_loader",
            &[],
        ).with(
            GltfSceneLoaderSystem::default(),
            "gltf_loader",
            &["scene_loader", "map_loader"],
        ).with(
            FPSRotationRhusicsSystem::<String, String>::new(0.3, 0.3),
            "free_rotation",
            &[],
        ).with(MouseFocusUpdateSystem::new(), "mouse_focus", &[])
        .with(CursorHideSystem::new(), "cursor_hide", &[])
        .with(GroundCheckerSystem::new(vec![ObjectType::Scene]), "ground_checker", &[])
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
        ).with(GravitySystem, "gravity", &[])
        .with_bundle(TransformBundle::new().with_dep(&[]))?
        .with(ContactSystem::default(), "contacts", &["bhop_movement"])
        .with(UiUpdaterSystem, "gameplay_ui_updater", &[])
        //.with(UiAutoTextSystem::<RelativeTimer>::default(), "timer_text_update", &[])
        .with_bundle(
            InputBundle::<String, String>::new().with_bindings_from_file(&key_bindings_path)?,
        )?.with_bundle(UiBundle::<String, String>::new())?
        .with_barrier()
        .with_bundle(DefaultPhysicsBundle3::<ObjectType>::new().with_spatial())?
        .with_bundle(RenderBundle::new(pipe, Some(display_config)))?;
    let mut game = Application::build(resources_directory, InitState::default())?
        .with_resource(asset_loader)
        .with_resource(AssetLoaderInternal::<FontAsset>::new())
        .build(game_data)?;
    game.run();
    Ok(())
}
